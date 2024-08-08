import os
import warnings

import torch
import cvcuda
import nvcv
import einops
from huggingface_hub import hf_hub_download
from torchvision.transforms import v2

# Input: torch.Tensor
# Output: torch.Tensor
def HWC3(x):
    assert x.dtype == torch.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return x.repeat(1, 1, 3)
    if C == 4:
        color = x[:, :, 0:3].float()
        alpha = x[:, :, 3:4].float() / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clamp(0, 255).to(torch.uint8)
        return y

# Input: torch.Tensor
# Output: torch.Tensor
def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(torch.round(torch.tensor(H / 64.0))) * 64
    W = int(torch.round(torch.tensor(W / 64.0))) * 64

    # Define the target size
    target_size = (H, W, C)

    return cvcuda_resize(input_image, target_size, cvcuda.Interp.LANCZOS if k > 1 else cvcuda.Interp.AREA) 

# Input: torch.Tensor
# Output: torch.Tensor
def cvcuda_resize(input_tensor, target_size, interpolation):
    # Create a CVCUDA tensor from the PyTorch tensor
    cvcuda_input_tensor = cvcuda.as_tensor(input_tensor, nvcv.TensorLayout.HWC)

    # Perform the resize operation on the GPU using CVCUDA
    resized_tensor = cvcuda.resize(cvcuda_input_tensor, target_size, interp=interpolation)

    # Convert the CVCUDA tensor back to a PyTorch tensor
    resized_tensor = torch.as_tensor(resized_tensor.cuda(), device="cuda:0")
    
    return resized_tensor

def safe_step(x, step=2):
    y = x.float() * (step + 1)
    y = y.int().float() / step
    return y

class DoubleConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel, layer_number):
        super().__init__()
        self.convs = torch.nn.Sequential()
        self.convs.append(torch.nn.Conv2d(in_channels=input_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        for i in range(1, layer_number):
            self.convs.append(torch.nn.Conv2d(in_channels=output_channel, out_channels=output_channel, kernel_size=(3, 3), stride=(1, 1), padding=1))
        self.projection = torch.nn.Conv2d(in_channels=output_channel, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0)

    def __call__(self, x, down_sampling=False):
        h = x
        if down_sampling:
            h = torch.nn.functional.max_pool2d(h, kernel_size=(2, 2), stride=(2, 2))
        for conv in self.convs:
            h = conv(h)
            h = torch.nn.functional.relu(h)
        return h, self.projection(h)


class ControlNetHED_Apache2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Parameter(torch.zeros(size=(1, 3, 1, 1)))
        self.block1 = DoubleConvBlock(input_channel=3, output_channel=64, layer_number=2)
        self.block2 = DoubleConvBlock(input_channel=64, output_channel=128, layer_number=2)
        self.block3 = DoubleConvBlock(input_channel=128, output_channel=256, layer_number=3)
        self.block4 = DoubleConvBlock(input_channel=256, output_channel=512, layer_number=3)
        self.block5 = DoubleConvBlock(input_channel=512, output_channel=512, layer_number=3)

    def __call__(self, x):
        h = x - self.norm
        h, projection1 = self.block1(h)
        h, projection2 = self.block2(h, down_sampling=True)
        h, projection3 = self.block3(h, down_sampling=True)
        h, projection4 = self.block4(h, down_sampling=True)
        h, projection5 = self.block5(h, down_sampling=True)
        return projection1, projection2, projection3, projection4, projection5

class HEDCudadetector:
    def __init__(self, netNetwork):
        self.netNetwork = netNetwork

    @classmethod
    def from_pretrained(cls, pretrained_model_or_path, filename=None, cache_dir=None, local_files_only=False):
        filename = filename or "ControlNetHED.pth"

        if os.path.isdir(pretrained_model_or_path):
            model_path = os.path.join(pretrained_model_or_path, filename)
        else:
            model_path = hf_hub_download(pretrained_model_or_path, filename, cache_dir=cache_dir, local_files_only=local_files_only)

        netNetwork = ControlNetHED_Apache2()
        netNetwork.load_state_dict(torch.load(model_path, map_location='cpu'))
        netNetwork.float().eval()

        return cls(netNetwork)
    
    def to(self, device, dtype=torch.float16):
        self.netNetwork.to(device, dtype=dtype)
        return self

    def compile(self):
        self.netNetwork = torch.compile(self.netNetwork, mode="reduce-overhead")
    
    def __call__(self, input_image, detect_resolution=512, image_resolution=512, safe=False, output_type="pil", scribble=False, **kwargs):
        if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
        if type(output_type) is bool:
            warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
            if output_type:
                output_type = "pil"

        # input_image is a 1CHW torch.Tensor
        # 1CHW -> CHW dtype=uint8
        input_image = input_image.squeeze(0).to(dtype=torch.uint8)

        # For some reason rearranging the torch.Tensor and passing it to
        # resize_image does not work and results in the following error:
        # RuntimeError: NVCV_ERROR_INVALID_ARGUMENT: Pitch of dimension 2 must be == 1 (packed), but it is 262144
        # But, converting the torch.Tensor into a nvcv.Tensor, reformatting from CHW to HWC and converting
        # back into a torch.Tensor before passing to resize_image works...
        # input_image = einops.rearrange(input_image, "c h w -> h w c")

        input_image = cvcuda.as_tensor(input_image, cvcuda.TensorLayout.CHW)
        # CHW -> HWC
        input_image = cvcuda.reformat(input_image, cvcuda.TensorLayout.HWC)
        # HWC torch.Tensor
        input_image = torch.as_tensor(input_image.cuda(), device="cuda:0")

        input_image = HWC3(input_image)
        input_image = resize_image(input_image, detect_resolution)

        assert input_image.ndim == 3
        H, W, C = input_image.shape
        with torch.no_grad():
            image_hed = einops.rearrange(input_image, 'h w c -> 1 c h w')
            edges = self.netNetwork(image_hed)
            edges = [einops.rearrange(e, "1 1 h w -> h w 1").to(dtype=torch.float32) for e in edges]
            edges = [cvcuda_resize(e, (H, W, 1), interpolation=cvcuda.Interp.LINEAR) for e in edges]
            edges = torch.stack(edges, dim=2)
            edge = 1 / (1 + torch.exp(-torch.mean(edges, dim=2).to(dtype=torch.float64)))
            if safe:
                edge = safe_step(edge)
            edge = (edge * 255.0).clip(0, 255).to(dtype=torch.uint8)

        detected_map = edge
        detected_map = HWC3(detected_map)

        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cvcuda_resize(detected_map, (H, W, C), interpolation=cvcuda.Interp.LINEAR)
        detected_map = einops.rearrange(detected_map, "h w c -> c h w")
        
        # TODO
        # if scribble:
        #     detected_map = nms(detected_map, 127, 3.0)
        #     detected_map = cv2.GaussianBlur(detected_map, (0, 0), 3.0)
        #     detected_map[detected_map > 4] = 255
        #     detected_map[detected_map < 255] = 0

        if output_type == "pil":
            detected_map = v2.Compose([v2.ToPILImage()])(detected_map)

        return detected_map