import os

import cv2 as cv
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from apilib.util import app, validate_uploads, get_weights, get_weights_for_request, get_pytorch_device, colorize_probabilitymap

from config import device
from data_gen import data_transforms, get_alpha, gen_trimap
from utils import ensure_folder

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

IMG_FOLDER = 'input'

if __name__ == '__main__':
    checkpoint = 'BEST_checkpoint.tar'
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model = checkpoint['model'].module
    model = model.to(device)
    model.eval()

    transformer = data_transforms['valid']

    ensure_folder('input')
    ensure_folder('output')

    files = [f for f in os.listdir(IMG_FOLDER) if f.endswith('.png')]

    for file in tqdm(files):
        filename = os.path.join(IMG_FOLDER, file)
        img = cv.imread(filename)
        print(img.shape)
        h, w = img.shape[:2]

        x = torch.zeros((1, 4, h, w), dtype=torch.float)
        image = img[..., ::-1]  # RGB
        image = transforms.ToPILImage()(image)
        image = transformer(image)
        x[0:, 0:3, :, :] = image

        filename = os.path.join('input', file)
        print('reading {}...'.format(filename))
        im_alpha = get_alpha(filename)
        trimap = gen_trimap(im_alpha)

        x[0:, 3, :, :] = torch.from_numpy(trimap.copy() / 255.)
            # print(torch.max(x[0:, 3, :, :]))
            # print(torch.min(x[0:, 3, :, :]))
            # print(torch.median(x[0:, 3, :, :]))

            # Move to GPU, if available
        x = x.type(torch.FloatTensor).to(device)

        with torch.no_grad():
            pred = model(x)

        pred = pred.cpu().numpy()
        pred = pred.reshape((h, w))

        pred[trimap == 0] = 0.0
        pred[trimap == 255] = 1.0

        out = (pred.copy() * 255).astype(np.uint8)

        filename = os.path.join('output', file)
        cv.imwrite(filename, out)
        print('wrote {}.'.format(filename))





@app.route('/pytorch/getmatting', methods=['GET', 'POST'])
def getmatting():

    weight_files = get_weights('/weights/HairPSPNet/', '.pth')

    if request.method == 'POST':
        try:
            (source,) = [validate_uploads(request, f) for f in ['source']]
        except ValueError as err:
            flash(str(err))
            return redirect(request.url)

        weights = get_weights_for_request('weights', weight_files)
        net, test_image_transforms = get_gethairmask_models(weights)
        
        source_img_np = cv2.imread(source[1])
        source_img = Image.fromarray(source_img_np)

        with torch.no_grad():
            net.eval()
            data = test_image_transforms(source_img).to(get_pytorch_device()).unsqueeze(0)
            logit = net(data)
            pred = torch.sigmoid(logit.cpu())[0][0].data.numpy()

        mh, mw = data.size(2), data.size(3) # rows, cols
        iw, ih = source_img.size            # width, height
        delta_h = mh - ih
        delta_w = mw - iw
        top = delta_h // 2
        bottom = mh - (delta_h - top)
        left = delta_w // 2
        right = mw - (delta_w - left)

        pred=pred[top:bottom, left:right]

        return_option = request.form.get('return', 'numpy')

        if return_option == 'numpy':
            npy_io = BytesIO()
            np.save(npy_io, pred)

            npy_io.seek(0)
            return send_file(npy_io, mimetype='application/octet-stream', as_attachment=True, attachment_filename='{}.hair-mask.npy'.format(source[0]))

        if return_option == 'overlay':
            overlay = colorize_probabilitymap(pred)

            source_img_np = source_img_np * 0.5 +  overlay * 0.5

            img_io = BytesIO()
            img_bytes = cv2.imencode('.png', source_img_np)[1].tobytes()
            img_io.write(img_bytes)
            img_io.seek(0)

            return send_file(img_io, mimetype='image/png', as_attachment=True, attachment_filename='{}.hair-mask.png'.format(source[0]))

    return '''

