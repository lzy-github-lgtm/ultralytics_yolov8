# pip install grad-cam -i https://pypi.tuna.tsinghua.edu.cn/simple
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, cv2, os, shutil
import numpy as np


np.random.seed(3407)
from PIL import Image
from ultralytics.nn.tasks import DetectionModel as Model
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM, GradCAMElementWise, GradCAMPlusPlus, HiResCAM, LayerCAM,
                              RandomCAM, ScoreCAM, XGradCAM)


def letterbox(im, new_shape=(640, 640), color=(255, 182, 193), auto=True, scaleFill=False, scaleup=True, stride=32):
    # 调整大小并填充图像，同时满足步幅多重约束
    shape = im.shape[:2]  # 当前形状 [高, 宽]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 缩放比例 (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # 仅缩小，不放大 (用于更好的验证 mAP)
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r  # 宽度、高度比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh填充
    if auto:  # 最小矩形
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh填充
    elif scaleFill:  # 拉伸
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # 宽度、高度比例

    dw /= 2  # 将填充分为2侧
    dh /= 2

    if shape[::-1] != new_unpad:  # 调整大小
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # 添加边框
    return im, ratio, (dw, dh)


class yolov8_cam:
    def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
        # 初始化yolov8_heatmap类
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()  # 将检查点的state_dict转换为FP32
        model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # 交集
        model.load_state_dict(csd, strict=False)  # 加载
        model.eval()

        target_layers = [eval(layer)]
        method = eval(method)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(int)
        self.__dict__.update(locals())

    def post_process(self, result):
        # 后处理模型输出
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], \
               torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]], \
               xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()

    def draw_detections(self, box, color, name, img):
        # 绘制检测结果框
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2) # 绘制目标框
        # cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
        #             lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_path, save_path):
        # 如果存在目录，则删除
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        # 如果不存在目录，则创建
        os.makedirs(save_path, exist_ok=True)

        # 图像处理
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)

        # 初始化ActivationsAndGradients
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

        # 获取ActivationsAndResult
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        # 后处理为yolo输出
        post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
        for i in range(int(post_result.size(0) * self.ratio)):
            if float(post_result[i].max()) < self.conf_threshold:
                break

            self.model.zero_grad()
            # 获取此预测的最大概率
            if self.backward_type == 'class' or self.backward_type == 'all':
                score = post_result[i].max()
                score.backward(retain_graph=True)

            if self.backward_type == 'box' or self.backward_type == 'all':
                for j in range(4):
                    score = pre_post_boxes[i, j]
                    score.backward(retain_graph=True)

            # 处理热图
            if self.backward_type == 'class':
                gradients = grads.gradients[0]
            elif self.backward_type == 'box':
                gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
            else:
                gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + \
                            grads.gradients[4]
            b, k, u, v = gradients.size()
            weights = self.method.get_cam_weights(self.method, None, None, None, activations,
                                                  gradients.detach().numpy())
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            if (saliency_map_max - saliency_map_min) == 0:
                continue
            saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

            # 添加热图和框到图像
            cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
            cam_image = self.draw_detections(post_boxes[i], (255, 182, 193),
                                             f'{self.model_names[int(post_result[i, :].argmax())]} {float(post_result[i].max()):.2f}',
                                             cam_image)
            cam_image = Image.fromarray(cam_image)
            cam_image.save(f'{save_path}/{i}.png')


def get_params():
    # 定义默认参数
    params = {
        'weight': '/home/ac522/lzy/ultralytics_yolov8/runs/detect/train2/weights/best.pt',
        'cfg': 'ultralytics/cfg/models/v8/yolov8-AFPN.yaml',
        'device': 'cuda:0',
        'method': 'GradCAM',  # EigenCAM, EigenGradCAM, GradCAM, GradCAMElementWise, GradCAMPlusPlus, HiResCAM, LayerCAM, RandomCAM, ScoreCAM, XGradCAM)
        'layer': 'model.model[-4]',
        'backward_type': 'all',  # class, box, all
        'conf_threshold': 0.6,  # 0.6
        'ratio': 0.02  # 0.02-0.1
    }
    return params


if __name__ == '__main__':
    model = yolov8_cam(**get_params())
    model(
        '/home/ac522/lzy/ultralytics_yolov8/my_dataset/hardDetectImg/N-0-33-bolthead.jpg', # 目标图像，报错请使用绝对路径
        './cam_results') # 保存的文件位置