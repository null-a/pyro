import numpy as np
import torch
from PIL import Image, ImageDraw

from transform import window_to_image, over
from utils import batch_expand

def frames_to_rgb_list(frames):
    return frames[:, 0:3].data.numpy().tolist()

def img_to_arr(img):
    assert img.mode == 'RGBA'
    channels = 4
    w, h = img.size
    arr = np.fromstring(img.tobytes(), dtype=np.uint8)
    return arr.reshape(w * h, channels).T.reshape(channels, h, w)

def draw_rect(size, color):
    img = Image.new('RGBA', (size, size))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, size - 1, size - 1], outline=color)
    return torch.from_numpy(img_to_arr(img).astype(np.float32) / 255.0)

def draw_window_outline(cfg, z_where, color):
    n = z_where.size(0)
    rect = draw_rect(cfg.window_size, color)
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

def frames_to_tensor(arr):
    # Turn an array of frames (of length seq_len) returned by the
    # model into a (batch, seq_len, rest...) tensor.
    return torch.cat([t.unsqueeze(0) for t in arr]).transpose(0, 1)

def latents_to_tensor(xss):
    return torch.stack([torch.stack(xs) for xs in xss]).transpose(2, 0)

def arr_to_img(nparr):
    assert nparr.shape[0] == 4 # required for RGBA
    shape = nparr.shape[1:]
    return Image.frombuffer('RGBA', shape, (nparr.transpose((1,2,0)) * 255).astype(np.uint8).tostring(), 'raw', 'RGBA', 0, 1)

def save_frames(frames, path_fmt_str, offset=0):
    n = frames.shape[0]
    assert_size(frames, (n, 4, frames.size(2), frames.size(3)))
    for i in range(n):
        arr_to_img(frames[i].data.numpy()).save(path_fmt_str.format(i + offset))
