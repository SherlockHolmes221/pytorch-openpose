import sys

sys.path.insert(0, 'python')
import cv2
from body import Body
import os
import argparse
import json


def detect_for_hico_dataset(image_path, out_path):
    if not os.path.exists(image_path):
        assert False
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print(image_path)
    body_estimation = Body('model/body_pose_model.pth')
    image_names = os.listdir(image_path)
    print(len(image_names))

    for test_image in image_names:
        assert test_image.endswith(".jpg")
        print(test_image)
        save_path = test_image[:-4] + '_keypoints.json'
        if os.path.exists(os.path.join(out_path, save_path)):
            print("continue")
            continue
        oriImg = cv2.imread(os.path.join(image_path, test_image))  # B,G,R order
        assert oriImg is not None
        candidate, subset = body_estimation(oriImg)
        pose_per_image = []

        print("person num:", len(subset))
        for n in range(len(subset)):
            list = []

            for i in range(18):
                index = int(subset[n][i])
                if index == -1:
                    list.append(0)
                    list.append(0)
                    list.append(0)
                else:
                    list.append(candidate[index][0])
                    list.append(candidate[index][1])
                    list.append(candidate[index][2])
            pose_per_image.append({
                "pose_keypoints": list
            })

        print("save_path", save_path)
        with open(os.path.join(out_path, save_path), 'w') as f:
            json.dump(pose_per_image, f, indent=4)

        # canvas = copy.deepcopy(oriImg)
        # canvas = util.draw_bodypose(canvas, candidate, subset)
        # plt.imshow(canvas[:, :, [2, 1, 0]])
        # plt.axis('off')
        # plt.show()


def detect_for_vcoco_dataset(image_dir, out_path, data_name="train"):
    if not os.path.exists(image_dir):
        assert False
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    assert data_name == 'train' or data_name == 'val' or data_name == 'test'

    ids = json.load(open("vcoco_split_id/split_ids.json", 'r'))
    ids = ids[data_name]
    print(len(ids))
    body_estimation = Body('model/body_pose_model.pth')

    for id in ids:
        if data_name == 'train':
            image_name = "COCO_" + data_name + "2014_" + str(id).zfill(12) + ".jpg"
            image_path = os.path.join(image_dir, "train2014", image_name)
            assert os.path.exists(image_path)
        else:
            image_name1 = "COCO_" + "train" + "2014_" + str(id).zfill(12) + ".jpg"
            image_path1 = os.path.join(image_dir, "train2014", image_name1)
            image_name2 = "COCO_" + "val" + "2014_" + str(id).zfill(12) + ".jpg"
            image_path2 = os.path.join(image_dir, "val2014", image_name2)

            assert os.path.exists(image_path1) or os.path.exists(image_path2)

            if os.path.exists(image_path1):
                image_path = image_path1
                image_name = image_name1
            else:
                image_path = image_path2
                image_name = image_name2

        save_path = image_name[:-4] + '_keypoints.json'
        # if os.path.exists(os.path.join(out_path, save_path)):
        #     print("continue")
        #     continue
        oriImg = cv2.imread(image_path)  # B,G,R order
        assert oriImg is not None
        candidate, subset = body_estimation(oriImg)
        pose_per_image = []

        print("person num:", len(subset))
        for n in range(len(subset)):
            list = []

            for i in range(18):
                index = int(subset[n][i])
                if index == -1:
                    list.append(0)
                    list.append(0)
                    list.append(0)
                else:
                    list.append(candidate[index][0])
                    list.append(candidate[index][1])
                    list.append(candidate[index][2])
            pose_per_image.append({
                "pose_keypoints": list
            })

        print("save_path", save_path)
        with open(os.path.join(out_path, save_path), 'w') as f:
            json.dump(pose_per_image, f, indent=4)


# python body_dec.py --dataset vcoco --part_vcoco train --path /home/xian/media/data/coco/images/ --out_path /home/xian/Documents/code/my_no_frills/data_symlinks/coco_processed/human_pose/train2014
# python body_dec.py --dataset vcoco --part_vcoco val --path /home/xian/media/data/coco/images/ --out_path /home/xian/Documents/code/my_no_frills/data_symlinks/coco_processed/human_pose/val2014
# python body_dec.py --dataset vcoco --part_vcoco test --path /home/xian/media/data/coco/images/ --out_path /home/xian/Documents/code/my_no_frills/data_symlinks/coco_processed/human_pose/test2014
# python body_dec.py --dataset hico --path /home/xian/media/data/HIOC/hico_20160224_det/images/train2015 --out_path /home/xian/Documents/code/my_no_frills/data_symlinks/hico_processed/human_pose/train2015
# python body_dec.py --dataset hico --path /home/xian/media/data/HIOC/hico_20160224_det/images/test2015 --out_path /home/xian/Documents/code/my_no_frills/data_symlinks/hico_processed/human_pose/test2015
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset")
    parser.add_argument("--path")
    parser.add_argument("--out_path")
    parser.add_argument("--part_vcoco")
    args = parser.parse_args()

    assert args.dataset is not None
    assert args.path is not None
    assert args.out_path is not None

    if args.dataset == 'hico':
        detect_for_hico_dataset(args.path, args.out_path)
    elif args.dataset == 'vcoco':
        assert args.part_vcoco is not None
        detect_for_vcoco_dataset(args.path, args.out_path, args.part_vcoco)
    else:
        assert False
