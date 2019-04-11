from train import Train, Init, stylize
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--IMG_H", type=int, default=256)
parser.add_argument("--IMG_W", type=int, default=256)
parser.add_argument("--IMG_C", type=int, default=3)
parser.add_argument("--STYLE_H", type=int, default=512)
parser.add_argument("--STYLE_W", type=int, default=512)
parser.add_argument("--C_NUMS", type=int, default=10)#The number of style images
parser.add_argument("--BATCH_SIZE", type=int, default=2)
parser.add_argument("--LEARNING_RATE", type=float, default=0.001)
parser.add_argument("--CONTENT_WEIGHT", type=float, default=1.0)
parser.add_argument("--STYLE_WEIGHT", type=float, default=5.0)
parser.add_argument("--PATH_CONTENT", type=str, default="./MSCOCO//")
parser.add_argument("--PATH_STYLE", type=str, default="./style_imgs//")
parser.add_argument("--PATH_MODEL", type=str, default="./save_para//")#The path of the model's weights
parser.add_argument("--PATH_VGG16", type=str, default="./vgg_para//")#The path of pretrained model of VGG16
parser.add_argument("--IS_TRAINED", type=bool, default=True)


args = parser.parse_args()

if not args.IS_TRAINED:
    Train(IMG_H=args.IMG_H, IMG_W=args.IMG_W, IMG_C=args.IMG_C, STYLE_H=args.STYLE_H, STYLE_W=args.STYLE_W, C_NUMS=args.C_NUMS, batch_size=args.BATCH_SIZE, learning_rate=args.LEARNING_RATE,
      content_weight=args.CONTENT_WEIGHT, style_weight=args.STYLE_WEIGHT, path_content=args.PATH_CONTENT, path_style=args.PATH_STYLE,
      model_path=args.PATH_MODEL, vgg_path=args.PATH_VGG16)
else:
    parser.add_argument("--PATH_IMG", type=str, default="./imgs//zhang2.jpg")
    parser.add_argument("--PATH_RESULTS", type=str, default="./results//")
    parser.add_argument("--LABEL_1", type=int, default=9)#Style 1
    parser.add_argument("--LABEL_2", type=int, default=5)#Style 2
    parser.add_argument("--ALPHA", type=float, default=0.5)
    args = parser.parse_args()
    target, sess, content, y1, y2, alpha = Init(args.C_NUMS, args.PATH_MODEL)
    #alpha * style2 + (1 - alpha) * style1
    for a in range(11):
        args.ALPHA = a / 10
        stylize(args.PATH_IMG, args.PATH_RESULTS, args.LABEL_1, args.LABEL_2, args.ALPHA, target, sess, content, y1, y2, alpha)