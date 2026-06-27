# building a Multilayer perceptron 
class Neuron: 
    def __init__(self,nip):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nip)]
        self.b = Value(random.uniform(-1,1))

    def __call__(self,x):
        #wx+b
        activation = sum([wi*xi for wi,xi in zip(self.w,x)],self.b)
        out = activation.tanh()
        return out
    def parameters(self):
        return self.w+[self.b]
        
class Layer:
    def __init__(self,nip,nout):
        self.neurons = [Neuron(nip) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs)==1 else outs
        
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP: 
    def __init__(self,nip,nouts):
        ipop = [nip]+nouts
        self.layers = [Layer(ipop[i],ipop[i+1]) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
