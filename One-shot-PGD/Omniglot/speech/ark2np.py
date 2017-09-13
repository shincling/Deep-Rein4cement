#!/usr/bin/python
# coding=utf-8
from __future__ import absolute_import
import struct
import logging
import numpy as np


def read_utt_data(ark_path_offset):
    ark_path, offset = ark_path_offset.split(':')
    offset = int(offset)
    return _read_utt_data(ark_path, offset)[:,:40]


def _readtoken(ark_read):
    tok = ""
    ch, = ark_read.read(1)
    while ch != ' ':
        tok = tok + ch
        ch, = ark_read.read(1)
    return tok


def _read_utt_data(ark_path, offset):
    try:
        with open(ark_path, 'rb') as ark_read_buffer:
            ark_read_buffer.seek(offset, 0)
            ark_read_buffer.read(1)
            header = _readtoken(ark_read_buffer)
            if header[0] != "B":
                print "Input .ark file is not binary " + ark_path;
                exit(1)
            if header == "BFM":
                m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
                n, cols = struct.unpack('<bi', ark_read_buffer.read(5))
                tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 4), dtype=np.float32)
                utt_mat = np.reshape(tmp_mat, (rows, cols))

            elif header == "BDM":
                m, rows = struct.unpack('<bi', ark_read_buffer.read(5))
                n, cols = struct.unpack('<bi', ark_read_buffer.read(5))
                tmp_mat = np.frombuffer(ark_read_buffer.read(rows * cols * 8), dtype=np.float64)
                tmp_mat = np.asarray(tmp_mat, dtype=np.float32)
                utt_mat = np.reshape(tmp_mat, (rows, cols))
            elif header == "BCM":
                g_min_value, g_range, g_num_rows, g_num_cols = struct.unpack('ffii', ark_read_buffer.read(16))
                utt_mat = np.zeros([g_num_rows, g_num_cols], dtype=np.float32)
                # uint16 percentile_0; uint16 percentile_25; uint16 percentile_75; uint16 percentile_100;
                per_col_header = []
                for i in xrange(g_num_cols):
                    per_col_header.append(struct.unpack('HHHH', ark_read_buffer.read(8)))
                    # print per_col_header[i]

                tmp_mat = np.frombuffer(ark_read_buffer.read(g_num_rows * g_num_cols), dtype=np.uint8)

                pos = 0
                for i in xrange(g_num_cols):
                    p0 = float(g_min_value + g_range * per_col_header[i][0] / 65535.0)
                    p25 = float(g_min_value + g_range * per_col_header[i][1] / 65535.0)
                    p75 = float(g_min_value + g_range * per_col_header[i][2] / 65535.0)
                    p100 = float(g_min_value + g_range * per_col_header[i][3] / 65535.0)

                    d1 = float((p25 - p0) / 64.0)
                    d2 = float((p75 - p25) / 128.0)
                    d3 = float((p100 - p75) / 63.0)
                    for j in xrange(g_num_rows):
                        c = tmp_mat[pos]
                        if c <= 64:
                            utt_mat[j][i] = p0 + d1 * c
                        elif c <= 192:
                            utt_mat[j][i] = p25 + d2 * (c - 64)
                        else:
                            utt_mat[j][i] = p75 + d3 * (c - 192)
                        pos += 1


            elif header == "BCM2":
                g_min_value, g_range, g_num_rows, g_num_cols = struct.unpack('ffii', ark_read_buffer.read(16))

                tmp_mat = np.frombuffer(ark_read_buffer.read(g_num_rows * g_num_cols * 2), dtype=np.int16)
                utt_mat = np.asarray(tmp_mat, dtype=np.float32)
                utt_mat = utt_mat * g_range / 65535.0 + g_min_value

            elif header == "BFV":
                s, size = struct.unpack('<bi', ark_read_buffer.read(5))
                utt_mat = np.frombuffer(ark_read_buffer.read(size * 4), dtype=np.float32)

            elif header == "BDV":
                s, size = struct.unpack('<bi', ark_read_buffer.read(5))
                utt_mat = np.frombuffer(ark_read_buffer.read(size * 8), dtype=np.float64)
                utt_mat = np.asarray(utt_mat, dtype=np.float32)
            else:
                print "Invalid .ark file Format " + self.scp_data[index][0];
                exit(1)
            return utt_mat
    except Exception, e:
        print Exception, ":", e
        raise e
        # traceback.print_exc()
        # print 'Open Fea File Failed ', self.scp_data[index][0]
    return None


if __name__ == "__main__":
    ark_path_offset = "/home/sw/Shin/fisher/fisher_16k/pitchlog/raw_fbank_pitch_fisher_16k.1.ark:30327"
    ret = read_utt_data(ark_path_offset)
    print ret.shape
    logging.info(ret)
