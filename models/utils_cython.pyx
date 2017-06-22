import numpy as np
cimport numpy as cnp
import cython
from libc.string cimport memset
from libc.math cimport pow, abs
from cpython cimport array
import array
import numbers

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b


@cython.overflowcheck(False) # turn off bounds-checking for entire function
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)  # turn off negative index wrapping for entire function
def merge_patches_uint8(
        cnp.uint8_t[:, :, :] out_img,
        cnp.uint8_t[:, :, :, :] patches,
        int patches_cols, patch_size, int stride):

    cdef int oh = out_img.shape[0]
    cdef int ow = out_img.shape[1]
    cdef int oc = out_img.shape[2]
    cdef int pw
    cdef int ph
    if isinstance(patch_size, numbers.Number):
        pw = ph = patch_size
    else:
        pw = patch_size[0]
        ph = patch_size[1]
    oh = (oh - ph) / stride * stride + ph
    ow = (ow - pw) / stride * stride + pw
    cdef int patches_rows = patches.shape[0] / patches_cols
    cdef int y, x
    cdef int pi, pj
    cdef int py, px
    cdef int pjl, pju
    cdef int pil, piu
    cdef int[:] agg = array.array('i', [0] * oc)
    cdef int agg_count
    cdef int c
    for y in range(0, oh):
        pjl = int_max((y - ph) / stride + 1, 0)
        pju = int_min(y / stride + 1, patches_rows)
        for x in range(0, ow):
            pil = int_max((x - pw) / stride + 1,  0)
            piu = int_min(x / stride + 1, patches_cols)
            memset(&agg[0], 0, oc * sizeof(cnp.int32_t))
            agg_count = 0
            for pj in range(pjl, pju):
                for pi in range(pil, piu):
                    px = x - pi * stride
                    py = y - pj * stride
                    for c in range(oc):
                        agg[c] = agg[c] + patches[pi + pj * patches_cols][py, px, c]
                    agg_count += 1
            for c in range(oc):
                out_img[y, x, c] = <cnp.uint8_t>(agg[c] / agg_count)


@cython.overflowcheck(False) # turn off bounds-checking for entire function
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)  # turn off negative index wrapping for entire function
def merge_patches_float32(
        cnp.float32_t[:, :, :] out_img,
        cnp.float32_t[:, :, :, :] patches,
        int patches_cols, patch_size, int stride):

    cdef int oh = out_img.shape[0]
    cdef int ow = out_img.shape[1]
    cdef int oc = out_img.shape[2]
    cdef int pw
    cdef int ph
    if isinstance(patch_size, numbers.Number):
        pw = ph = patch_size
    else:
        pw = patch_size[0]
        ph = patch_size[1]
    oh = (oh - ph) / stride * stride + ph
    ow = (ow - pw) / stride * stride + pw
    cdef int patches_rows = patches.shape[0] / patches_cols
    cdef int y, x
    cdef int pi, pj
    cdef int py, px
    cdef int pjl, pju
    cdef int pil, piu
    cdef double[:] agg = array.array('d', [0] * oc)
    cdef int agg_count
    cdef int c
    #cdef double temp
    for y in range(0, oh):
        pjl = int_max((y - ph) / stride + 1, 0)
        pju = int_min(y / stride + 1, patches_rows)
        for x in range(0, ow):
            pil = int_max((x - pw) / stride + 1,  0)
            piu = int_min(x / stride + 1, patches_cols)
            #memset(&agg[0], 0, oc * sizeof(cnp.float64_t))
            for c in range(oc):
                agg[c] = 0.0
            agg_count = 0
            for pj in range(pjl, pju):
                for pi in range(pil, piu):
                    px = x - pi * stride
                    py = y - pj * stride
                    for c in range(oc):
                        agg[c] = agg[c] + patches[pi + pj * patches_cols][py, px, c]
                        #temp = patches[pi + pj * patches_cols][py, px, c]
                        #if temp < 0:
                        #    temp = 0.0
                        #agg[c] *= temp
                    agg_count += 1
            for c in range(oc):
                out_img[y, x, c] = <cnp.float32_t>(agg[c] / agg_count)
                #out_img[y, x, c] = <cnp.float32_t>(pow(agg[c], 1.0 / agg_count))
