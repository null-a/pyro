from dynair6 import DynAIR, load_data, frames_to_rgb_list, overlay_window_outlines, latent_seq_to_tensor

import torch

import numpy as np
from matplotlib import pyplot as plt

import pyro.poutine as poutine

def show_seq(seq):
    plt.figure(figsize=(8, 2))
    for r in range(2):
        for c in range(7):
            plt.subplot(2, 7, r * 7 + c + 1)
            plt.axis('off')
            plt.imshow(seq[r * 7 + c].transpose((1,2,0)))

X = load_data()

dynair = DynAIR(True,
                False,
                True,
                True)

dynair.load_state_dict(torch.load('dynair6.pytorch', map_location=lambda storage, loc: storage))

# vis.images(list(reversed(frames_to_rgb_list(X[ix+k].cpu()))), nrow=7)


# 2 is a extrap fail on appearance too
# 4 (fairly clearly) shows failure to model shade well

# 6 does show object in extrap.

ix = 4

# Show input.
seq = np.array(list(reversed(frames_to_rgb_list(X[ix]))))
show_seq(seq)
plt.savefig('/Users/paul/Downloads/input.png')

#plt.show()



# Show recon.

trace = poutine.trace(dynair.guide).get_trace(X[ix:ix+1])
frames, zs = poutine.replay(dynair.model, trace)(X[ix:ix+1], do_likelihood=False)

frames = latent_seq_to_tensor(frames)
zs = latent_seq_to_tensor(zs)

out = overlay_window_outlines(dynair, frames[0], zs[0, :, 0:2])

show_seq(np.array(frames_to_rgb_list(out)))
#vis.images(frames_to_rgb_list(out.cpu()), nrow=7)

#plt.show()
plt.savefig('/Users/paul/Downloads/recon.png')


# Extrapolate.

ex = X[ix:ix+1]
zs, y, w = dynair.guide(ex)
bkg = dynair.model_generate_bkg(w)

z = zs[-1]
frames = []
extrap_zs = []
for t in range(14):
    #z = dynair.model_transition(14 + t, z)
    z, _ = dynair.transition(z)
    frame_mean = dynair.model_emission(w, y, z, bkg)
    frames.append(frame_mean)
    extrap_zs.append(z)

extrap_frames = latent_seq_to_tensor(frames)
extrap_zs = latent_seq_to_tensor(extrap_zs)
out = overlay_window_outlines(dynair, extrap_frames[0], extrap_zs[0, :, 0:2])

show_seq(np.array(frames_to_rgb_list(out)))
plt.savefig('/Users/paul/Downloads/extra.png')
#plt.show()


# Also save background alone:

plt.figure()
plt.axis('off')
plt.imshow(bkg.data.numpy()[0,0:3].transpose((1,2,0)))
plt.savefig('/Users/paul/Downloads/bkg.png')

# Show object in several positions:




def show_obj_at(xx, yy):
    test_z = torch.cat((torch.FloatTensor([[xx, yy]]), zs[0][:,2:].data), 1)

    # Hmm. Seems like the latent state is encoding something to do
    # with appearance. Perhaps explains why extrapolation doesn't
    # work.

    #test_z = torch.FloatTensor([[xx, yy, -1, -1]])
    return dynair.model_emission(w, y, test_z, bkg)



# plt.figure()

# grd = [1.5, 0.75, 0, -0.75, -1.5]
# g = len(grd)
# for yi, yy in enumerate(grd):
#     for xi, xx in enumerate(grd):
#         plt.subplot(g, g, yi * g + xi + 1)
#         plt.axis('off')
#         plt.imshow(show_obj_at(xx, yy).data.numpy()[0,0:3].transpose((1,2,0)))

# plt.show()
