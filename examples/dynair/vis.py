import numpy as np
import torch
from PIL import Image, ImageDraw

from transform import window_to_image, over
from utils import batch_expand

def img_to_arr(img):
    #assert img.mode == 'RGBA'
    #channels = 4 if img.mode == 'RGBA' else 3
    if img.mode == 'RGBA':
        channels = 4
    elif img.mode == 'RGB':
        channels = 3
    elif img.mode == 'LA':
        channels = 2
    elif img.mode == 'L':
        channels = 1
    else:
        raise Exception('unsupport image mode')
    w, h = img.size
    arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    return arr.reshape(w * h, channels).T.reshape(channels, h, w)

def draw_rect(num_chan, size, color):
    assert num_chan == 3 or num_chan == 1
    mode = 'RGBA' if num_chan == 3 else 'LA'
    img = Image.new(mode, (size, size))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size - 1, size - 1], outline=color)
    return torch.from_numpy(img_to_arr(img).astype(np.float32) / 255.0)

def draw_window_outline(cfg, z_where, color):
    n = z_where.size(0)
    rect = draw_rect(cfg.num_chan, cfg.window_size, color)
    if z_where.is_cuda:
        rect = rect.cuda()
    rect_batch = batch_expand(rect.contiguous().view(-1), n).contiguous()
    return window_to_image(cfg, z_where, rect_batch)

def overlay_window_outlines(cfg, frame, z_where, color):
    return over(draw_window_outline(cfg, z_where, color), frame)

def overlay_window_outlines_conditionally(cfg, frame, z_where, color, ii):
    batch_size = z_where.size(0)
    presence_mask = ii.view(-1, 1, 1, 1)
    borders = batch_expand(torch.Tensor([-0.08, 0, 0]), batch_size).type_as(ii)
    return over(draw_window_outline(cfg, borders, color) * presence_mask,
                over(draw_window_outline(cfg, z_where, color) * presence_mask,
                     frame))

def overlay_multiple_window_outlines(cfg, frame, ws, obj_count):
    acc = frame
    for i in range(obj_count):
        acc = overlay_window_outlines(cfg, acc, ws[i], ['red', 'green', 'blue'][i % 3])
    return acc

def arr_to_img(nparr):
    assert nparr.shape[0] == 4 # required for RGBA
    shape = nparr.shape[1:]
    return Image.frombuffer('RGBA', shape, (nparr.transpose((1,2,0)) * 255).astype(np.uint8).tostring(), 'raw', 'RGBA', 0, 1)

def save_frames(frames, path_fmt_str, offset=0):
    n = frames.shape[0]
    assert_size(frames, (n, 4, frames.size(2), frames.size(3)))
    for i in range(n):
        arr_to_img(frames[i].data.numpy()).save(path_fmt_str.format(i + offset))
