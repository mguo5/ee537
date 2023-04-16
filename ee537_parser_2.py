import numpy as np
from scipy.linalg import lu_factor, lu_solve

class LUSolver:
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def lu_solve(self):
        lu, piv = lu_factor(self.A)
        x = lu_solve((lu, piv), self.b)

        return x

class Circuit:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.unique_voltages = []

    def create_lhs(self):
        self.lhs = np.zeros((self.i, self.j))

    """
        Dealing wiht the Yn part of the matrix
    """

    def resistor_update_lhs_zero(self, value, i, j):
        self.lhs[i - 1][j - 1] += 1/value

    def resistor_update_lhs(self, value, i, j):
        self.lhs[i - 1][i - 1] += 1/value
        self.lhs[j - 1][j - 1] += 1/value
        self.lhs[i - 1][j - 1] += -1/value
        self.lhs[j - 1][i - 1] += -1/value 

    def voltage_control_current_update_lhs(self, value, i, j, k, l):
        if(i != 0 and k == 0):
            self.lhs[i - 1][l - 1] = -1 * value
        if(i != 0 and l == 0):
            self.lhs[i - 1][k - 1] = value
        if(j != 0 and k == 0):
            self.lhs[j - 1][l - 1] = value
        if(j != 0 and l == 0):
            self.lhs[j - 1][k - 1] = -1 * value
        if(i != 0 and j != 0 and l != 0 and k != 0):
            self.lhs[i - 1][k - 1] = value
            self.lhs[i - 1][l - 1] = -1 * value
            self.lhs[j - 1][k - 1] = -1 * value
            self.lhs[j - 1][l - 1] = value

    def return_lhs(self):
        print(self.lhs)
        return self.lhs

    """
        Dealing with Independent Voltages for LHS
    """
    def independent_volt_update(self, code, i, j, value):
        if(code in self.unique_voltages):
            index = self.unique_voltages.index(code) + self.i - 1
        else:
            self.unique_voltages.append(code)
            index = self.unique_voltages.index(code) + self.i - 1
        
        if(i == 0):
            self.lhs[j - 1][index] = -1
            self.lhs[index][j - 1] = -1
        if(j == 0):
            self.lhs[i - 1][index] = 1
            self.lhs[index][i - 1] = 1
        if(i != 0 and j != 0):
            self.lhs[i - 1][index] = 1
            self.lhs[j - 1][index] = -1
            self.lhs[index][i - 1] = 1
            self.lhs[index][j - 1] = -1

        self.rhs[index][0] = value

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
    num_volt = 1
    save_matrix_txt = False

    c = Circuit(node_amount + num_volt, node_amount + num_volt)
    c.create_lhs()
    c.create_rhs()

    resistors = []
    voltages = []
    currents = []
    vccs = []

    with open("test_file.txt", "r") as netlist_file:
        # Read the contents of the file into a string variable
        netlist = netlist_file.read()
        lines = netlist.split("\n")
        for line in lines:
            if line.strip() != "":
                parts = line.split()
                element = None
                node_values = None
                element_values = None
                if(len(parts) == 4):
                    element = parts[0]
                    nodes_values = parts[1:3]
                    element_value = parts[3]
                elif(len(parts) == 6):
                    element = parts[0]
                    nodes_values = parts[1:5]
                    element_value = parts[5]
                elif(len(parts) == 1):
                    element = parts[0]
                
                if element.startswith("R"):
                    resistors.append([element] + nodes_values + [element_value])
                elif element.startswith("V"):
                    voltages.append([element] + nodes_values + [element_value])
                elif element.startswith("I"):
                    currents.append([element] + nodes_values + [element_value])
                elif element.startswith("G"):
                    vccs.append([element] + nodes_values + [element_value])
                elif element.startswith(".text"):
                    save_matrix_txt = True

    for r in resistors:
        code = r[0]
        np = int(r[1])
        nm = int(r[2])
        value = int(r[3])

        if(np == 0):
            c.resistor_update_lhs_zero(value, nm, nm)
        elif(nm == 0):
            c.resistor_update_lhs_zero(value, np, np)
        else:
            c.resistor_update_lhs(value, np, nm)
    
    for v in voltages:
        code = v[0]
        np = int(v[1])
        nm = int(v[2])
        value = int(v[3])

        c.independent_volt_update(code, np, nm, value)

    for i in currents:
        code = i[0]
        np = int(i[1])
        nm = int(i[2])
        value = int(i[3])

        if(np == 0):
            c.i_current_update_rhs_zero(value, nm, nm)
        elif(nm == 0):
            c.i_current_update_rhs_zero(value, np, np)
        else:
            c.i_current_update_rhs(value, np, nm)

    for g in vccs:
        code = g[0]
        np = int(g[1])
        nm = int(g[2])
        ncp = int(g[3])
        ncm = int(g[4])
        value = int(g[5])

        c.voltage_control_current_update_lhs(value, np, nm, ncp, ncm)
    
    print("=======LHS Matrix======")
    lhs = c.return_lhs()
    print("=======RHS Matrix======")
    rhs = c.return_rhs()

    l = LUSolver(lhs, rhs)
    print(l.lu_solve())