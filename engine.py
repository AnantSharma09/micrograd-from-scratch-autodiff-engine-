# Building Value data structure
class Value:
    
    def __init__(self,data,op='',_children=()):
        self.data = data
        self.op = ''
        self._backward_recipe = lambda:None
        self._prev = set(_children)
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data})"
        
    def __add__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data+other.data)
        out.op = '+'
        out._prev = {self,other}

        def _backward_recipe():
            self.grad += 1.0*out.grad
            other.grad += 1.0*out.grad

        out._backward_recipe=_backward_recipe
            
        return out 

    def __mul__(self,other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data*other.data)
        out.op = '*'
        out._prev = {self,other}

        def _backward_recipe():
            self.grad += other.data*out.grad
            other.grad += self.data*out.grad

        out._backward_recipe=_backward_recipe
            
        return out

    def __pow__(self,other):
        assert isinstance(other,(int,float)),"only supporting int/float"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward_recipe():
            out.grad += (other * self.data**(other-1)) * out.grad
        out._backward_recipe = _backward_recipe

        return out 
        
    def tanh(self):
        x = self.data
        t = (math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
        out = Value(t)
        out.op = 'tanh'
        out._prev = {self}

        def _backward_recipe():
            self.grad += (1-out.data**2)*out.grad

        out._backward_recipe = _backward_recipe
        return out 

    def exp(self):
         out = Value(math.exp(self.data))
         out._prev = {self}
         out.op = 'exp'

         def _backward_recipe():
             self.grad += out.data*out.grad
         self._backward_recipe= _backward_recipe
         return out

        
    def backward_movement(self):
        visited = set()
        topo = []

        def build_topo(node):
            if node in visited:
                return 
            visited.add(node)
            for parent in node._prev:
                build_topo(parent)
                
            topo.append(node)

        build_topo(self)
        self.grad=1
        for node in reversed(topo):
            node._backward_recipe()
            
    def __neg__(self): #-self
        return self*-1

    def __radd__(self,other): #other + self
         return self + other

    def __sub__(self,other):
        return self+(-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
