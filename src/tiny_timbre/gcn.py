"""From-scratch GCN layer, GCN head over the genre graph, and the flat multi-label head.

GCN layer:  H^(l+1) = ReLU(A_hat H^(l) W^(l))   (no ReLU after the last layer)
GCN head:   H^0 = I (one-hot), two layers C -> 512 -> 256, scores z = H^(L) x
Flat head:  Linear 256 -> C

Owner: Ayesha
"""
