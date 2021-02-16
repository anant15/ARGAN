import argparse
import os
import io
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms
import PIL.Image as pil_image
import glob
import time
import random

from models import GeneratorResNet

cudnn.benchmark = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean = [0.3572, 0.3610, 0.3702]
std = [0.2033, 0.2016, 0.2011]

transforms_evaluation = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, required=False)
    parser.add_argument('--image_path', type=str, required=False)
    parser.add_argument('--outputs_dir', type=str, required=False)
    parser.add_argument('--jpeg_quality', type=int, default=40)
    parser.add_argument('--input_dir', type=str, required=False)

    opt, unknown = parser.parse_known_args()
    model = GeneratorResNet()

    #opt.weights_path = "/nfs/nas2VehiScan/IMPData/Video_Analytics/anant-seatbelt/argan-output/act-17-less-GGAN-600-patch_48-lr_g0.0001-lr_d1e-05-quality_40/weights/generator_599.pth"

    state_dict = model.state_dict()
    for n, p in torch.load(opt.weights_path, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model = model.to(device)
    print(device)
    model.eval()

    if opt.input_dir:
        filenames = [os.path.join(opt.input_dir, file) for file in os.listdir(opt.input_dir) if file.endswith(("ppm", "jpeg", "png", "jpg"))]
        print(filenames)
    else:
        filenames = opt.image_path

    if not os.path.exists(opt.outputs_dir):
        os.makedirs(opt.outputs_dir)

    avg_time = []
    for filename in filenames:
        print("file is", filename)
        input = pil_image.open(filename).convert('RGB')
        print("Input size:", input.size)

        #input = transforms.ToTensor()(input).unsqueeze(0).to(device)
        input = transforms_evaluation(input).unsqueeze(0).to(device)
        output_path = os.path.join(opt.outputs_dir, '{}-{}.ppm'.format(filename.split("/")[-1].split(".")[0], "argan"))

        with torch.no_grad():
            start = time.time()
            pred = model(input)[-1]
            end = time.time()
            print("Time for one image", end-start)
            avg_time.append(end-start)

        #pred = (pred - pred.min().item()) / (pred.max().item() - pred.min().item())
        pred = pred * torch.tensor(std).unsqueeze(-1).unsqueeze(-1).to(device) + torch.tensor(mean).unsqueeze(-1).unsqueeze(-1).to(device)

        print(pred.min(), pred.max())
        pred = pred.mul_(255.0).clamp_(0.0, 255.0).squeeze(0).permute(1, 2, 0).byte().cpu().numpy()
        output = pil_image.fromarray(pred, mode='RGB')
        print("Output size", output.size)
        print("Output dir is", opt.outputs_dir)
        output.save(output_path)
        #print(os.path.join(opt.outputs_dir, '{}_{}.png'.format(filename, "EDAR")))
        print("Output saved")
    print(sum(avg_time[1:])/len(avg_time[1:]))

#         input = input.resize((912, 505))
        #rand_num = random.randrange(1, 4)
#         print(rand_num)
#         if rand_num == 1:
#             input = input.resize((897, 486))
#         elif rand_num == 2:
#             input = input.resize((2048, 1536))
#         else:
#             input = input.resize((int(2048/2), int(1536/2)))
#         print("Input resized:", input.size)
