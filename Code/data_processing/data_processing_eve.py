import os
import cv2

# 定义train文件夹路径
train_folder = '/data/wyl_data/eve_dataset/train'
imgpath="/data/wyl_data/EVE/Image"

if not os.path.exists(imgpath):
    os.makedirs(imgpath)

# 定义MP4文件列表
mp4_files = [
    'webcam_c_face.mp4',
    'webcam_l_face.mp4',
    'webcam_r_face.mp4'
]

trainfiles = os.listdir(train_folder)
trainfiles.sort()

# 遍历train文件夹下的各个子文件夹
for train_folder_name in trainfiles:
    train_folder_path = os.path.join(train_folder, train_folder_name)

    # 检查是否为文件夹
    if not os.path.isdir(train_folder_path):
        continue

    stepfiles = os.listdir(train_folder_path)
    stepfiles.sort()

    # 遍历每个子文件夹下的step子文件夹
    for step_folder_name in stepfiles:
        step_folder_path = os.path.join(train_folder_path, step_folder_name)

        # 检查是否为文件夹
        if not os.path.isdir(step_folder_path):
            continue

        step=step_folder_name.split('_')[0]

        # 遍历MP4文件列表
        for mp4_file in mp4_files:
            # 拼接MP4文件路径
            mp4_file_path = os.path.join(step_folder_path, mp4_file)

            # 检查MP4文件是否存在
            if not os.path.isfile(mp4_file_path):
                continue

            # 解析输出路径
            mp4_file_parts = mp4_file.split('.')[0].split('_')
            output_subpath = os.path.join(mp4_file_parts[0]+'_'+mp4_file_parts[1], mp4_file_parts[2])

            # 创建保存图像的文件夹路径
            save_folder_path = os.path.join(imgpath, train_folder_name, output_subpath)
            os.makedirs(save_folder_path, exist_ok=True)

            # 打开视频文件
            video_capture = cv2.VideoCapture(mp4_file_path)

            # 检查视频文件是否成功打开
            if not video_capture.isOpened():
                print(f"无法打开视频文件: {mp4_file_path}")
                continue

            # 逐帧读取视频并保存为JPEG图像文件
            frame_count = 1
            while True:
                # 读取视频的一帧
                ret, frame = video_capture.read()

                # 检查视频是否已经读取完毕
                if not ret:
                    break
                if (frame_count-1)%15==0:
                    # height, width, _ = frame.shape
                    # center = width // 2
                    #
                    # # 沿中线裁剪成两张图像
                    # left_image = frame[:, :center]
                    # right_image = frame[:, center:]
                    # out_file1 = os.path.join(save_folder_path, "left", f"{step}_{frame_count}.jpg")
                    # out_file2 = os.path.join(save_folder_path, "right", f"{step}_{frame_count}.jpg")
                    #
                    # # 保存为JPEG图像
                    # cv2.imwrite(out_file1, left_image)
                    # cv2.imwrite(out_file2, right_image)

                    # 生成图像文件路径
                    output_file = os.path.join(save_folder_path, f"{step}_{frame_count}.jpg")

                    # 保存帧为JPEG图像文件
                    cv2.imwrite(output_file, frame)

                    print(f"保存帧 {frame_count} 为 {output_file}")

                # 增加帧计数器
                frame_count += 1

            # 释放视频对象
            video_capture.release()
