cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.set cimport set
from libcpp.map cimport map
from libcpp.map cimport pair
from cython.operator cimport postincrement as postincrement
from cython.operator cimport dereference as dereference
from definitions cimport Tensor as CTensor, Blob as CBlob, Layer as CLayer, shared_ptr, LayerParameter, ApolloNet

import numpy as pynp
import h5py
import os
import caffe_pb2
import sys

np.import_array()
cdef public api tonumpyarray(float* data, long long size) with gil:
    #if not (data and size >= 0): raise ValueError
    cdef np.npy_intp dims = size
    #NOTE: it doesn't take ownership of `data`. You must free `data` yourself
    return np.PyArray_SimpleNewFromData(1, &dims, np.NPY_FLOAT, <void*>data)

cdef extern from "caffe/caffe.hpp" namespace "caffe::Caffe":
    void set_random_seed(unsigned int)
    enum Brew:
        CPU = 0
        GPU = 1
    void set_mode(Brew)
    void SetDevice(int)
    void set_logging_verbosity(int level)

cdef class Caffe:
    def __cinit__(self):
        pass
    @staticmethod
    def set_random_seed(seed):
        set_random_seed(seed)
    @staticmethod
    def set_device(device_id):
        SetDevice(device_id)
    @staticmethod
    def set_mode_cpu():
        set_mode(CPU)
    @staticmethod
    def set_mode_gpu():
        set_mode(GPU)
    @staticmethod
    def set_logging_verbosity(level):
        set_logging_verbosity(level)


cdef extern from "caffe/layer_factory.hpp" namespace "caffe::LayerRegistry<float>":
    cdef shared_ptr[CLayer] CreateLayer(LayerParameter& param)

cdef class Layer(object):
    cdef shared_ptr[CLayer] thisptr
    def __cinit__(self):
        pass
    cdef void Init(self, shared_ptr[CLayer] other):
        self.thisptr = other
    property layer_param:
        def __get__(self):
            param = caffe_pb2.LayerParameter()
            cdef string s
            self.thisptr.get().layer_param().SerializeToString(&s)
            param.ParseFromString(s)
            return param
    property buffers:
        def __get__(self):
            buffers = []
            cdef vector[shared_ptr[CBlob]] cbuffers
            (&cbuffers)[0] = self.thisptr.get().buffers()
            for i in range(cbuffers.size()):
                new_blob = Blob()
                new_blob.Init(cbuffers[i])
                buffers.append(new_blob)
            return buffers
    property params:
        def __get__(self):
            params = []
            cdef vector[shared_ptr[CBlob]] cparams
            (&cparams)[0] = self.thisptr.get().blobs()
            for i in range(cparams.size()):
                new_blob = Blob()
                new_blob.Init(cparams[i])
                params.append(new_blob)
            return params

cdef class Tensor:
    cdef shared_ptr[CTensor] thisptr
    def __cinit__(self):
        self.thisptr.reset(new CTensor())
    cdef void Init(self, shared_ptr[CTensor] other):
        self.thisptr = other
    cdef void AddFrom(Tensor self, Tensor other):
        self.thisptr.get().AddFrom(other.thisptr.get()[0])
    cdef void MulFrom(Tensor self, Tensor other):
        self.thisptr.get().MulFrom(other.thisptr.get()[0])
    cdef void AddMulFrom(Tensor self, Tensor other, float alpha):
        self.thisptr.get().AddMulFrom(other.thisptr.get()[0], alpha)
    cdef void CopyFrom(Tensor self, Tensor other):
        self.thisptr.get().CopyFrom(other.thisptr.get()[0])
    def reshape(self, pytuple):
        cdef vector[int] shape
        for x in pytuple:
            shape.push_back(x)
        self.thisptr.get().Reshape(shape)
    property shape:
        def __get__(self):
            return self.thisptr.get().shape()
    def count(self):
        return self.thisptr.get().count()
    property mem:
        def __get__(self):
            result = tonumpyarray(self.thisptr.get().mutable_cpu_mem(),
                        self.thisptr.get().count())
            sh = self.shape
            result.shape = sh if len(sh) > 0 else (1,)
            return pynp.copy(result)
        def __set__(self, value):
            if hasattr(value, 'shape'):
                result = tonumpyarray(self.thisptr.get().mutable_cpu_mem(),
                            self.thisptr.get().count())
                sh = self.shape
                result.shape = sh if len(sh) > 0 else (1,)
                result[:] = value
            else:
                self.thisptr.get().SetValues(value)
    def copy_from(self, other):
        self.CopyFrom(other)
    def axpy(self, other, alpha):
        self.AddMulFrom(other, alpha)
    def __iadd__(self, other):
        self.AddFrom(other)
        return self
    def __isub__(self, other):
        self.AddMulFrom(other, -1.)
    def __imul__(self, other):
        if type(other) == type(self):
            self.MulFrom(other)
        else:
            self.thisptr.get().scale(other)
        return self
    def set_mem(self, other):
        if type(self) == type(other):
            self.CopyFrom(other)
        else:
            self.thisptr.get().SetValues(other)


cdef class Blob(object):
    cdef shared_ptr[CBlob] thisptr
    def __cinit__(self):
        pass
    cdef void Init(self, shared_ptr[CBlob] other):
        self.thisptr = other
    def count(self):
        return self.thisptr.get().count()
    property shape:
        def __get__(self):
            return self.thisptr.get().shape()
    property diff:
        def __get__(self):
            result = tonumpyarray(self.thisptr.get().mutable_cpu_diff(),
                        self.thisptr.get().count())
            sh = self.shape
            result.shape = sh if len(sh) > 0 else (1,)
            return result
    property data:
        def __get__(self):
            result = tonumpyarray(self.thisptr.get().mutable_cpu_data(),
                        self.thisptr.get().count())
            sh = self.shape
            result.shape = sh if len(sh) > 0 else (1,)
            return result

    property diff_tensor:
        def __get__(self):
            cdef shared_ptr[CTensor] ctensor = self.thisptr.get().diff()
            diff = Tensor()
            diff.Init(ctensor)
            return diff
        def __set__(self, other):
            self.diff_tensor.copy_from(other)

    property data_tensor:
        def __get__(self):
            cdef shared_ptr[CTensor] ctensor = self.thisptr.get().data()
            data = Tensor()
            data.Init(ctensor)
            return data
        def __set__(self, other):
            self.data_tensor.copy_from(other)

        
cdef class Net:
    cdef ApolloNet* thisptr
    def __cinit__(self, phase='train'):
        self.thisptr = new ApolloNet()
        if phase == 'train':
            self.thisptr.set_phase_train()
        elif phase == 'test':
            self.thisptr.set_phase_test()
        else:
            assert False, "phase must be one of ['train', 'test']"
    def __dealloc__(self):
        del self.thisptr
    def forward(self, arch):
        return arch.forward(self)
    def forward_layer(self, layer):
        return self.thisptr.ForwardLayer(layer.p.SerializeToString(), layer.r.SerializeToString())
    def backward_layer(self, layer_name):
        self.thisptr.BackwardLayer(layer_name)
    def backward(self):
        for layer_name in self.active_layer_names()[::-1]:
            self.backward_layer(layer_name)
    def update(self, lr, momentum=0., clip_gradients=-1, weight_decay=0.):
        diffnorm = self.diff_l2_norm() 
        clip_scale = 1.
        if clip_gradients > 0:
            if diffnorm > clip_gradients:
                clip_scale = clip_gradients / diffnorm
        params = self.params
        for param_name in self.active_param_names():
            self.update_param(params[param_name],
                              lr * clip_scale * self.param_lr_mults(param_name),
                              momentum,
                              weight_decay * self.param_decay_mults(param_name))
        self.reset_forward()
    def update_param(self, param, lr, momentum, weight_decay):
        param.diff_tensor.axpy(param.data_tensor, weight_decay)
        param.data_tensor.axpy(param.diff_tensor, -lr)
        param.diff_tensor *= momentum
    def diff_l2_norm(self):
        return self.thisptr.DiffL2Norm()
    def reset_forward(self):
        self.thisptr.ResetForward()
    def active_layer_names(self):
        cdef vector[string] layer_names
        layer_names = self.thisptr.active_layer_names()
        return layer_names
    def active_param_names(self):
        cdef set[string] param_set
        (&param_set)[0] = self.thisptr.active_param_names()
        cdef set[string].iterator it = param_set.begin()
        cdef set[string].iterator end = param_set.end()
        param_names = []
        while it != end:
            param_names.append(dereference(it))
            postincrement(it)
        return param_names
    def param_lr_mults(self, name):
        cdef map[string, float] lr_mults
        (&lr_mults)[0] = self.thisptr.param_lr_mults()
        return lr_mults[name]

    def param_decay_mults(self, name):
        cdef map[string, float] decay_mults
        (&decay_mults)[0] = self.thisptr.param_decay_mults()
        return decay_mults[name]

    property layers:
        def __get__(self):
            cdef map[string, shared_ptr[CLayer]] layers_map
            (&layers_map)[0] = self.thisptr.layers()

            layers = {}
            cdef map[string, shared_ptr[CLayer]].iterator it = layers_map.begin()
            cdef map[string, shared_ptr[CLayer]].iterator end = layers_map.end()
            cdef string layer_name
            cdef shared_ptr[CLayer] layer
            while it != end:
                layer_name = dereference(it).first
                layer = dereference(it).second
                py_layer = Layer()
                py_layer.Init(layer)
                layers[layer_name] = py_layer
                postincrement(it)

            return layers

    property params:
        def __get__(self):
            cdef map[string, shared_ptr[CBlob]] param_map
            (&param_map)[0] = self.thisptr.params()

            blobs = {}
            cdef map[string, shared_ptr[CBlob]].iterator it = param_map.begin()
            cdef map[string, shared_ptr[CBlob]].iterator end = param_map.end()
            cdef string blob_name
            cdef shared_ptr[CBlob] blob_ptr
            while it != end:
                blob_name = dereference(it).first
                blob_ptr = dereference(it).second
                new_blob = Blob()
                new_blob.Init(blob_ptr)
                blobs[blob_name] = new_blob
                postincrement(it)

            return blobs

    property blobs:
        def __get__(self):
            cdef map[string, shared_ptr[CBlob]] blob_map
            (&blob_map)[0] = self.thisptr.blobs()

            blobs = {}
            cdef map[string, shared_ptr[CBlob]].iterator it = blob_map.begin()
            cdef map[string, shared_ptr[CBlob]].iterator end = blob_map.end()
            cdef string blob_name
            cdef shared_ptr[CBlob] blob_ptr
            while it != end:
                blob_name = dereference(it).first
                blob_ptr = dereference(it).second
                new_blob = Blob()
                new_blob.Init(blob_ptr)
                blobs[blob_name] = new_blob
                postincrement(it)

            return blobs

    def save(self, filename):
        assert filename.endswith('.h5'), "saving only supports h5 files"
        with h5py.File(filename, 'w') as f:
            for name, value in self.params.items():
                f[name] = pynp.copy(value.data)

    def load(self, filename):
        if len(self.params) == 0:
            sys.stderr.write('WARNING, loading into empty net.')
        _, extension = os.path.splitext(filename)
        if extension == '.h5':
            with h5py.File(filename, 'r') as f:
                params = self.params
                for name, stored_value in f.items():
                    if name in params:
                        params[name].data[:] = stored_value
        elif extension == '.caffemodel':
            self.thisptr.CopyTrainedLayersFrom(filename)
        else:
            assert False, "Error, filename is neither h5 nor caffemodel: %s, %s" % (filename, extension)
    def copy_params_from(self, other):
        self_params = self.params
        if len(self_params) == 0:
            sys.stderr.write('WARNING, copying into empty net.')
        for name, value in other.params.items():
            if name in self_params:
                self_params[name].data[:] = pynp.copy(value.data)


cdef extern from "caffe/proto/caffe.pb.h" namespace "caffe":
    cdef cppclass NumpyDataParameter:
        void add_data(float data)
        void add_shape(unsigned int shape)
        string DebugString()
    cdef cppclass RuntimeParameter:
        NumpyDataParameter* mutable_numpy_data_param()
        NumpyDataParameter numpy_data_param()
        bool SerializeToString(string*)
        string DebugString()

class PyRuntimeParameter(object):
    def __init__(self, result):
        self.result = result
    def SerializeToString(self):
        return self.result

def make_numpy_data_param(numpy_array):
    assert numpy_array.dtype == pynp.float32
    cdef vector[int] v
    for x in numpy_array.shape:
        v.push_back(x)
    cdef string s = make_numpy_data_param_fast(pynp.ascontiguousarray(numpy_array.flatten()), v)
    return PyRuntimeParameter(str(s)) #s.encode('utf-8'))

cdef string make_numpy_data_param_fast(np.ndarray[np.float32_t, ndim=1] numpy_array, vector[int] v):
    cdef RuntimeParameter runtime_param
    cdef NumpyDataParameter* numpy_param
    numpy_param = runtime_param.mutable_numpy_data_param()
    cdef int length = len(numpy_array)
    cdef int i
    for i in range(v.size()):
        numpy_param[0].add_shape(v[i])
    for i in range(length):
        numpy_param[0].add_data(numpy_array[i])
    cdef string s
    runtime_param.SerializeToString(&s)
    return s