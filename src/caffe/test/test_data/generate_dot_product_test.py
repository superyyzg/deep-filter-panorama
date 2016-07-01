import numpy as np
import caffe


def write_as_proto(fname, mat):
    blob = caffe.io.array_to_blobproto(mat)
    open(fname, 'wb').write(blob.SerializeToString())


def random_VM_case():
    mat_a = np.random.rand(2, 5).astype(np.float32)
    mat_b = np.random.rand(2, 5, 4).astype(np.float32)
    mat_c = np.zeros((2, 4), dtype=np.float32)

    for i in range(2):
        mat_c[i] = np.dot(mat_a[i], mat_b[i])

    write_as_proto('src/caffe/test/test_data/dot_prodoct_a_vm.binaryproto',
                   mat_a)
    write_as_proto('src/caffe/test/test_data/dot_prodoct_b_vm.binaryproto',
                   mat_b)
    write_as_proto('src/caffe/test/test_data/dot_prodoct_c_vm.binaryproto',
                   mat_c)


if __name__ == '__main__':
    random_VM_case()
