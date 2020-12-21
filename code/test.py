import os
import time
import torch
import torchvision
from PIL import Image
import numpy as np

from dataloader import populate_chunk_list, single_chunk
from network import RNNmodule

def test(args, file_list=None):
    with torch.no_grad():
        network = RNNmodule(3, args.h_size, args.num_adj, (args.kernel, args.kernel), False).cuda()
        network.load_state_dict(torch.load('%s%s_Epoch%d.pth' % (args.snapshots_path, args.model_name, args.epochs-1)))
        print('load %s%s_Epoch%d.pth' % (args.snapshots_path, args.model_name, args.epochs-1))
        network.eval()

        img_seq = file_list
        hiddens = None
        for i, file in enumerate(img_seq):
            if not file_list:
                img = Image.open(args.validate_img + file)
            else:
                img = Image.open(file)
            img = (np.asarray(img) / 255.0)
            imgs = torch.from_numpy(img).permute(2, 0, 1).float()
            imgs = imgs.cuda().unsqueeze(0)

            start = time.time()
            adj_img, adj_param, hiddens = network(imgs, hiddens)
            end_time = (time.time() - start)
            print(file, end_time, adj_param)

            result_path = os.path.dirname(file).replace('image', 'img_corr')
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            if i<2:
                torchvision.utils.save_image(imgs, os.path.join(result_path, os.path.basename(file)))
            else:
                torchvision.utils.save_image(adj_img, os.path.join(result_path, os.path.basename(file)))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=0)
    parser.add_argument("--chunk", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument('--snapshots_path', type=str, default="snapshots/")
    parser.add_argument("--num_adj", type=int, default=1)
    parser.add_argument("--seq_num", type=int, default=6)
    parser.add_argument("--h_size", type=int, default=32)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--exp_patch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    chunk_list = single_chunk(args.chunk)  
    # chunk_list = populate_chunk_list('/home/zhangyb/projects/ColonData/sequences/')
    for chunk in chunk_list:
        # print(chunk[0].split('/')[-1])
        chunk.sort()
        test(args, chunk)
