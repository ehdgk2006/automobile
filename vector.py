from math import sin, cos


class Vector:
    def __init__(self, *values: float):
        self.v = list(values)
    

    def norm(self, n: int = 2):
        res = 0

        for i in self.v:
            res += i**n
        
        return res**(1/n)
    

    def transform(self, v: object) -> None:
        if len(self.v) == len(v.v):
            res = []
            
            for i in range(len(self.v)):
                res.append(self.v[i] + v.v[i])
            
            self.v = res
        else:
            raise ValueError(f"wrong dimension: {len(self.v)}d, {len(v.v)}d")
    

    def rotate(self, rad: float) -> None:
        x = self.v[0] * cos(rad) - self.v[1] * sin(rad)
        y = self.v[0] * sin(rad) + self.v[1] * cos(rad)

        self.v = [x, y]
    

    def scale(self, s: float) -> None:
        res = []
        
        for i in range(len(self.v)):
            res.append(self.v[i] * s)
        
        self.v = res

    
    def copy(self):
        return Vector(*self.v.copy())
    

    def __len__(self):
        return len(self.v)


    def __getitem__(self, idx: int):
        return self.v[idx]

    
    def __add__(self, o: object):
        if len(self.v) == len(o.v):
            res = []
            
            for i in range(len(self.v)):
                res.append(self.v[i] + o.v[i])
            
            return Vector(*res)
        else:
            raise ValueError(f"wrong dimension: {len(self.v)}d, {len(o.v)}d")
        
    
    def __sub__(self, o: object):
        if len(self.v) == len(o.v):
            res = []
            
            for i in range(len(self.v)):
                res.append(self.v[i] - o.v[i])
            
            return Vector(*res)
        else:
            raise ValueError(f"wrong dimension: {len(self.v)}d, {len(o.v)}d")
    
    
    def __mul__(self, o: float):
        res = []
        
        for i in range(len(self.v)):
            res.append(self.v[i] * o)
        
        return Vector(*res)
    
    
    def __truediv__(self, o: float):
        res = []
        
        for i in range(len(self.v)):
            res.append(self.v[i] / o)
        
        return Vector(*res)

    
    def __eq__(self, o: object):
        if len(self.v) == len(o.v):
            for i in range(len(self.v)):
                if self.v[i] != o.v[i]:
                    return False
                
            return True
        else:
            raise ValueError(f"wrong dimension: {len(self.v)}d, {len(o.v)}d")
