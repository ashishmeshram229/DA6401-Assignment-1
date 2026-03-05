import numpy as np
from ann.layer import Layer
from ann.losses import get_loss
from ann.optimizers import get_optimizer


class NeuralNetwork:

    def __init__(self, args):

        self.args = args

        hidden_sizes = getattr(args, "hidden_size", [128,128,128])
        activations = getattr(args, "activation", ["relu","relu","relu"])
        num_hidden = getattr(args, "num_layers", 3)

        if not isinstance(hidden_sizes,list):
            hidden_sizes=[hidden_sizes]

        if not isinstance(activations,list):
            activations=[activations]

        hidden_sizes=(hidden_sizes+[hidden_sizes[-1]]*num_hidden)[:num_hidden]
        activations=(activations+[activations[-1]]*num_hidden)[:num_hidden]

        self.loss_name=getattr(args,"loss","cross_entropy")
        self.loss_fn,self.loss_grad_fn=get_loss(self.loss_name)

        self.weight_decay=getattr(args,"weight_decay",0.0)
        weight_init=getattr(args,"weight_init","xavier")

        self.layers=[]

        in_dim=28*28

        for out_dim,act in zip(hidden_sizes,activations):

            layer=Layer(in_dim,out_dim,act,weight_init)
            layer.optimizer=get_optimizer(args)

            self.layers.append(layer)
            in_dim=out_dim

        output_layer=Layer(in_dim,10,"none",weight_init)
        output_layer.optimizer=get_optimizer(args)

        self.layers.append(output_layer)

    # ---------------------------------------------------------
    # forward
    # ---------------------------------------------------------

    def forward(self,x):

        out=np.asarray(x,dtype=np.float64)

        if out.ndim==1:
            out=out.reshape(1,-1)

        elif out.ndim>2:
            out=out.reshape(out.shape[0],-1)

        for layer in self.layers:
            out=layer.forward(out)

        return out

    # ---------------------------------------------------------
    # helpers
    # ---------------------------------------------------------

    def _labels(self,y,B,C):

        arr=np.asarray(y)

        if arr.ndim==2 and arr.shape[1]==C:
            arr=np.argmax(arr,axis=1)
        else:
            arr=arr.flatten()

        return np.clip(arr[:B].astype(int),0,C-1)

    def _one_hot(self,labels,C):

        oh=np.zeros((len(labels),C),dtype=np.float64)
        oh[np.arange(len(labels)),labels]=1

        return oh

    def _softmax(self,x):

        e=np.exp(x-np.max(x,axis=1,keepdims=True))
        return e/(np.sum(e,axis=1,keepdims=True)+1e-12)

    # ---------------------------------------------------------
    # loss
    # ---------------------------------------------------------

    def compute_loss(self,logits,y):

        if logits.ndim==1:
            logits=logits.reshape(1,-1)

        B,C=logits.shape

        labels=self._labels(y,B,C)
        y_oh=self._one_hot(labels,C)

        if self.loss_name=="cross_entropy":

            probs=self._softmax(logits)

            loss=-np.mean(np.sum(y_oh*np.log(probs+1e-9),axis=1))

        else:

            loss=np.mean(np.sum((logits-y_oh)**2,axis=1))

        return float(loss)

    # ---------------------------------------------------------
    # backward
    # ---------------------------------------------------------

    def backward(self,logits,y_true):

        if logits.ndim==1:
            logits=logits.reshape(1,-1)

        B,C=logits.shape

        labels=self._labels(y_true,B,C)
        y_oh=self._one_hot(labels,C)

        grad=self.loss_grad_fn(logits,y_oh)

        if grad is None:
            grad=np.zeros_like(logits)

        for layer in reversed(self.layers):

            grad=layer.backward(grad)

            if grad is None:
                grad=np.zeros((B,layer.W.shape[0]))

        return self.layers[0].grad_W,self.layers[0].grad_b

    # ---------------------------------------------------------
    # update
    # ---------------------------------------------------------

    def update(self,lr):

        for layer in self.layers:
            layer.update(lr,self.weight_decay)

    # ---------------------------------------------------------
    # save weights
    # ---------------------------------------------------------

    def get_weights(self):

        weights={}

        for i,l in enumerate(self.layers):

            weights[i]={
                "W":l.W.copy(),
                "b":l.b.copy()
            }

        return weights

    # ---------------------------------------------------------
    # universal weight loader
    # ---------------------------------------------------------

    def set_weights(self,weights):

        # unwrap numpy container
        if isinstance(weights,np.ndarray):

            if weights.dtype==object:
                weights=weights.item()

            else:
                weights=weights.tolist()

        pairs=[]

        def extract(obj):

            if isinstance(obj,dict):

                if "W" in obj and "b" in obj:
                    pairs.append((np.asarray(obj["W"]),np.asarray(obj["b"])))

                for v in obj.values():
                    extract(v)

            elif isinstance(obj,(list,tuple)):

                if len(obj)==2 and isinstance(obj[0],np.ndarray):
                    pairs.append((np.asarray(obj[0]),np.asarray(obj[1])))
                else:
                    for v in obj:
                        extract(v)

        extract(weights)

        if len(pairs)==0:
            raise ValueError("Unsupported weight format")

        # rebuild network
        self.layers=[]

        for i,(W,b) in enumerate(pairs):

            act="none" if i==len(pairs)-1 else "relu"

            layer=Layer(W.shape[0],W.shape[1],act,"zeros")

            layer.W=W.copy()
            layer.b=b.copy()

            layer.grad_W=np.zeros_like(W)
            layer.grad_b=np.zeros_like(b)

            layer.optimizer=get_optimizer(self.args)

            self.layers.append(layer)