import numpy as np

class Circuit:
    def __init__(self, i, j):
        self.i = i
        self.j = j

    """
        Dealing wiht the Yn part of the matrix
    """
    def create_yn(self):
        self.yn = np.zeros((self.i, self.j))

    def resistor_update_yn_zero(self, value, i, j):
        self.yn[i - 1][j - 1] += 1/value

    def resistor_update_yn(self, value, i, j):
        self.yn[i - 1][i - 1] += 1/value
        self.yn[j - 1][j - 1] += 1/value
        self.yn[i - 1][j - 1] += -1/value
        self.yn[j - 1][i - 1] += -1/value 

    def voltage_control_current_update_yn(self, value, i, j, k, l):
        if(i != 0 and k == 0):
            self.yn[i - 1][l - 1] = -1 * value
        if(i != 0 and l == 0):
            self.yn[i - 1][k - 1] = value
        if(j != 0 and k == 0):
            self.yn[j - 1][l - 1] = value
        if(j != 0 and l == 0):
            self.yn[j - 1][k - 1] = -1 * value
        if(i != 0 and j != 0 and l != 0 and k != 0):
            self.yn[i - 1][k - 1] = value
            self.yn[i - 1][l - 1] = -1 * value
            self.yn[j - 1][k - 1] = -1 * value
            self.yn[j - 1][l - 1] = value

    def return_yn(self):
        print(self.yn)
        return self.yn

    """
        Dealing with the RHS of the equation
    """
    def create_rhs(self):
        self.rhs = np.zeros((self.i, 1))

    def i_current_update_rhs_zero(self, value, i, j):
        self.rhs[i - 1][0] += value

    def i_current_update_rhs(self, value, i, j):
        self.rhs[i - 1][0] += value
        self.rhs[j - 1][0] -= value
    
    def return_rhs(self):
        print(self.rhs)
        return self.rhs


if __name__ == "__main__":
    node_amount = 3
    test_string = "R1   1   2   1000\nI1   0   1   2\nR2    2   3   1000\nR3    3   0   1000"
    lines = test_string.split("\n")

    c = Circuit(node_amount, node_amount)
    c.create_yn()
    c.create_rhs()

    array = []
    for line in lines:
        if line:
            row = line.split()
            array = row[:6]

            if array[0][0] == 'R':
                code = array[0]
                np = int(array[1])
                nm = int(array[2])
                value = int(array[3])

                if(np == 0):
                    c.resistor_update_yn_zero(value, nm, nm)
                elif(nm == 0):
                    c.resistor_update_yn_zero(value, np, np)
                else:
                    c.resistor_update_yn(value, np, nm)
            
            if array[0][0] == "I":
                code = array[0]
                np = int(array[1])
                nm = int(array[2])
                value = int(array[3])

                if(np == 0):
                    c.i_current_update_rhs_zero(value, nm, nm)
                elif(nm == 0):
                    c.i_current_update_rhs_zero(value, np, np)
                else:
                    c.i_current_update_rhs(value, np, nm)

            if array[0][0] == "G":
                code = array[0]
                np = int(array[1])
                nm = int(array[2])
                ncp = int(array[3])
                ncm = int(array[4])
                value = int(array[5])

                c.voltage_control_current_update_yn(value, np, nm, ncp, ncm)

    c.return_yn()
    c.return_rhs()