
import numpy as np

lhs = []
rhs = []

node_tracker = []

class Resistor:
    def __init__(self, code, n1, n2, value):
        self.code = str(code)
        self.n1 = int(n1)
        self.n2 = int(n2)
        self.value = int(value)
        node_tracker.append(self.n1)
        node_tracker.append(self.n2)

    def create_sub_matrix_string(self):
           # Check if connected to ground
        if(self.n1 == 0 or self.n2 == 0):
            sub_matrix = np.zeros((1, 1), dtype=np.dtype('a16'))
            sub_matrix[0][0] = '1/' + self.code
            self.sub_matrix_string = sub_matrix
            return sub_matrix
        
        # Check the difference between the nodes
        diff = abs(self.n2 - self.n1) + 1
        sub_matrix = np.zeros((diff, diff), dtype=np.dtype('a16'))
        sub_matrix[0][0] = '1/' + self.code
        sub_matrix[0][abs(self.n2 - self.n1)] = '-1/' + self.code
        sub_matrix[abs(self.n2 - self.n1)][0] = '-1/' + self.code
        sub_matrix[abs(self.n2 - self.n1)][abs(self.n2 - self.n1)] = '1/' + self.code
        self.sub_matrix_string = sub_matrix
        return sub_matrix
    
    def create_sub_matrix(self):
        # Check if connected to ground
        if(self.n1 == 0 or self.n2 == 0):
            sub_matrix = np.zeros((1, 1))
            sub_matrix[0][0] = 1/self.value
            self.sub_matrix = sub_matrix
            return sub_matrix
        
        # Check the difference between the nodes
        diff = abs(self.n2 - self.n1) + 1
        sub_matrix = np.zeros((diff, diff))
        sub_matrix[0][0] = 1/self.value
        sub_matrix[0][abs(self.n2 - self.n1)] = -1/self.value
        sub_matrix[abs(self.n2 - self.n1)][0] = -1/self.value
        sub_matrix[abs(self.n2 - self.n1)][abs(self.n2 - self.n1)] = 1/self.value
        self.sub_matrix = sub_matrix
        return sub_matrix

    def obtain_appropriate_node_val(self):
        if self.n1 == 0:
            return self.n2 - 1
        if self.n2 == 0:
            return self.n1 - 1
        if self.n1 > self.n2:
            return self.n2 - 1
        return self.n1 - 1


class LinearDependentSource:
    def __init__(self, g, e, f, h, code):
        self.code = code
        self.g = g
        self.e = e
        self.f = f
        self.h = h

class VoltageControlSource:
    def __init__(self, np, nm, ncp, ncm, value, code):
        self.code = str(code)    #Could be G or E
        self.np = int(np)
        self.nm = int(nm)
        self.ncp = int(ncp)
        self.ncm = int(ncm)
        self.value = int(value)
        node_tracker.append(self.np)
        node_tracker.append(self.nm)
        node_tracker.append(self.ncp)
        node_tracker.append(self.ncm)

    def create_sub_matrix_string(self):
        if("G" in self.code):
            if(self.nm == 0):
                if(self.ncp == 0 and self.ncm != 0):
                    sub_matrix = np.zeros((1, 1), dtype=np.dtype('a16'))
                    sub_matrix[0][0] = "-1" + self.code
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm == 0):
                    sub_matrix = np.zeros((1, 1), dtype=np.dtype('a16'))
                    sub_matrix[0][0] = self.code
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm != 0):
                    diff = abs(self.ncp - self.ncm) + 1
                    sub_matrix = np.zeros((1, diff), dtype=np.dtype('a16'))
                    sub_matrix[0][0] = self.code
                    sub_matrix[0][abs(self.ncp - self.ncm)] = "-1" + self.code
                    self.sub_matrix = sub_matrix
                    return sub_matrix
            elif(self.np == 0):
                if(self.ncp == 0 and self.ncm != 0):
                    sub_matrix = np.zeros((1, 1), dtype=np.dtype('a16'))
                    sub_matrix[0][0] = self.code
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm == 0):
                    sub_matrix = np.zeros((1, 1), dtype=np.dtype('a16'))
                    sub_matrix[0][0] = "-1" + self.code
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm != 0):
                    diff = abs(self.ncp - self.ncm) + 1
                    sub_matrix = np.zeros((1, diff), dtype=np.dtype('a16'))
                    sub_matrix[0][0] = "-1" + self.code
                    sub_matrix[0][abs(self.ncp - self.ncm)] = self.code
                    self.sub_matrix = sub_matrix
                    return sub_matrix
            else:
                diff = abs(self.np - self.nm) + 1
                if(self.ncp == 0 and self.ncm != 0):
                    sub_matrix = np.zeros((diff, 1), dtype=np.dtype('a16'))
                    sub_matrix[0][0] = "-1" + self.code
                    sub_matrix[abs(self.np - self.nm)][0] = self.code
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm == 0):
                    sub_matrix = np.zeros((diff, 1), dtype=np.dtype('a16'))
                    sub_matrix[0][0] = self.code
                    sub_matrix[abs(self.np - self.nm)][0] = "-1" + self.code
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm != 0):
                    sub_matrix = np.zeros((diff, diff), dtype=np.dtype('a16'))
                    sub_matrix[0][0] = self.code
                    sub_matrix[0][abs(self.ncp - self.ncm)] = "-1" + self.code
                    sub_matrix[0][abs(self.ncp - self.ncm)] = "-1" + self.code
                    sub_matrix[abs(self.ncp - self.ncm)][abs(self.ncp - self.ncm)] = self.code
                    self.sub_matrix = sub_matrix
                    return sub_matrix
        
        else:
            pass
    

    def create_sub_matrix(self):
        if("G" in self.code):
            if(self.nm == 0):
                if(self.ncp == 0 and self.ncm != 0):
                    sub_matrix = np.zeros((1, 1))
                    sub_matrix[0][0] = -1*self.value
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm == 0):
                    sub_matrix = np.zeros((1, 1))
                    sub_matrix[0][0] = self.value
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm != 0):
                    diff = abs(self.ncp - self.ncm) + 1
                    sub_matrix = np.zeros((1, diff))
                    sub_matrix[0][0] = self.value
                    sub_matrix[0][abs(self.ncp - self.ncm)] = -1*self.value
                    self.sub_matrix = sub_matrix
                    return sub_matrix
            elif(self.np == 0):
                if(self.ncp == 0 and self.ncm != 0):
                    sub_matrix = np.zeros((1, 1))
                    sub_matrix[0][0] = self.value
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm == 0):
                    sub_matrix = np.zeros((1, 1))
                    sub_matrix[0][0] = -1*self.value
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm != 0):
                    diff = abs(self.ncp - self.ncm) + 1
                    sub_matrix = np.zeros((1, diff))
                    sub_matrix[0][0] = -1*self.value
                    sub_matrix[0][abs(self.ncp - self.ncm)] = self.value
                    self.sub_matrix = sub_matrix
                    return sub_matrix
            else:
                diff = abs(self.np - self.nm) + 1
                if(self.ncp == 0 and self.ncm != 0):
                    sub_matrix = np.zeros((diff, 1))
                    sub_matrix[0][0] = -1*self.value
                    sub_matrix[abs(self.np - self.nm)][0] = self.value
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm == 0):
                    sub_matrix = np.zeros((diff, 1))
                    sub_matrix[0][0] = self.value
                    sub_matrix[abs(self.np - self.nm)][0] = -1*self.value
                    self.sub_matrix = sub_matrix
                    return sub_matrix
                if(self.ncp != 0 and self.ncm != 0):
                    sub_matrix = np.zeros((diff, diff))
                    sub_matrix[0][0] = self.value
                    sub_matrix[0][abs(self.ncp - self.ncm)] = -1*self.value
                    sub_matrix[0][abs(self.ncp - self.ncm)] = -1*self.value
                    sub_matrix[abs(self.ncp - self.ncm)][abs(self.ncp - self.ncm)] = self.value
                    self.sub_matrix = sub_matrix
                    return sub_matrix
        else:
            pass

class CurrentControlSource:
    def __init__(self, np, nm, name, value, code):
        self.code = code    #Could be F or H
        self.np = np
        self.nm = nm
        self.name = name
        self.value = value
        node_tracker.append(self.np)
        node_tracker.append(self.nm)

    def create_sub_matrix_string(self):
        if("F" in self.code):
            sub_matrix = np.zeros((2, 2), dtype=np.dtype('a16'))
            sub_matrix[0][0] = self.code
            sub_matrix[0][1] = "-1" + self.code
            sub_matrix[1][0] = "-1" + self.code
            sub_matrix[1][1] = self.code

            if(self.np == 0):
                sub_matrix[0][0] = 0
                sub_matrix[0][1] = 0
            if(self.nm == 0):
                sub_matrix[1][0] = 0
                sub_matrix[1][1] = 0
            if(self.ncp == 0):
                sub_matrix[0][0] = 0
                sub_matrix[1][0] = 0
            if(self.ncm == 0):
                sub_matrix[0][1] = 0
                sub_matrix[1][1] = 0
            self.sub_matrix_string = sub_matrix
            return sub_matrix
        
        else:
            pass
    
    def create_sub_matrix(self):
        if("F" in self.code):
            sub_matrix = np.zeros((2, 2))
            sub_matrix[0][0] = self.value
            sub_matrix[0][1] = -1*self.value
            sub_matrix[1][0] = -1*self.value
            sub_matrix[1][1] = self.value

            if(self.np == 0):
                sub_matrix[0][0] = 0
                sub_matrix[0][1] = 0
            if(self.nm == 0):
                sub_matrix[1][0] = 0
                sub_matrix[1][1] = 0
            if(self.ncp == 0):
                sub_matrix[0][0] = 0
                sub_matrix[1][0] = 0
            if(self.ncm == 0):
                sub_matrix[0][1] = 0
                sub_matrix[1][1] = 0
            self.sub_matrix = sub_matrix
            return sub_matrix
        
        else:
            pass

class IndependentSource:
    def __init__(self, np, nm, value, code):
        self.code = str(code)    #Could be V or I
        self.np = int(np)
        self.nm = int(nm)
        self.value = int(value)
        node_tracker.append(self.np)
        node_tracker.append(self.nm)

    def create_sub_matrix_string(self):
        if("I" in self.code):
            sub_matrix = np.zeros((2, 1), dtype=np.dtype('a16'))
            sub_matrix[0][0] = "-1" + self.code
            sub_matrix[1][0] = self.code

            if(self.np == 0):
                sub_matrix[0][0] = 0
            if(self.nm == 0):
                sub_matrix[1][0] = 0
            self.sub_matrix_string = sub_matrix
            return sub_matrix

        else:
            pass

    def create_sub_matrix(self):
        if("I" in self.code):
            sub_matrix = np.zeros((2, 1))
            sub_matrix[0][0] = -1*self.value
            sub_matrix[1][0] = self.value

            if(self.np == 0):
                sub_matrix[0][0] = 0
            if(self.nm == 0):
                sub_matrix[1][0] = 0
            self.sub_matrix = sub_matrix
            return sub_matrix

        else:
            pass

    def obtain_appropriate_node_val(self):
        if self.nm > self.np:
            return self.nm - 1
        else:
            return self.np - 1

class Circuit:
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def form_left_matrix(self):
        self.left_matrix = np.zeros((self.i, self.j))
    
    def form_right_matrix(self):
        self.right_matrix = np.zeros((self.i, 1))

    def compute_matrix_equivalent(self):
        for element in lhs:
            row_start = element.obtain_appropriate_node_val()
            col_start = element.obtain_appropriate_node_val()

            self.left_matrix[row_start:row_start+element.sub_matrix.shape[0], col_start:col_start+element.sub_matrix.shape[1]] += element.sub_matrix
        
        for element in rhs:
            row_start = element.obtain_appropriate_node_val()
            col_start = 0

            self.right_matrix[row_start:row_start+element.sub_matrix.shape[0], col_start:col_start+element.sub_matrix.shape[1]] += element.sub_matrix
        return self.left_matrix, self.right_matrix


def pretty_print():
    print(f"{lhs[0].create_sub_matrix_string()} = {rhs[0].create_sub_matrix_string()}")


if __name__ == "__main__":
    # test_string = "R1   1   0   1000\n"
    test_string = "R1   1   2   1000\nI1   0   1   2\nR2    2   3   1000\nR3    3   0   1000"
    lines = test_string.split("\n")

    array = []
    for line in lines:
        if line:
            row = line.split()
            array = row[:6]

        print(array)
        if array[0][0] == 'R':
            # print("This is a resistor")
            r = Resistor(array[0], array[1], array[2], array[3])
            r.create_sub_matrix()
            r.create_sub_matrix_string()
            lhs.append(r)
        elif array[0][0] in ['G', 'E']:
            print("This is a Voltage Controlled Source")
        elif array[0][0] in ['F', 'H']:
            print("This is a Current Controlled Source")
        elif array[0][0] in ['V', 'I']:
            # print("This is an Independent Source")
            s = IndependentSource(array[1], array[2], array[3], array[0])
            s.create_sub_matrix()
            s.create_sub_matrix_string()
            rhs.append(s)
            # print(s.create_sub_matrix())
            # print(s.create_sub_matrix_string())
        else:
            print("None")

    max_node_size = max(max(node_tracker), 2)
    c = Circuit(max_node_size, max_node_size)
    c.form_left_matrix()
    c.form_right_matrix()
    lhs_output, rhs_output = c.compute_matrix_equivalent()
    print(f"{lhs_output} = {rhs_output}")
    # pretty_print()

