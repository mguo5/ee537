import numpy as np
from scipy.linalg import lu_factor, lu_solve
import sys
import matplotlib.pyplot as plt


# TODO: Adding capacitor and inductors
# Adding capacitor is bascially a resistor in parallel with a current source.
# The current source will be changing for each delta_t, but the resistor will
# stay the same. Notice that because of this, the LHS will never be changed, 
# but there will be a lot of different RHS. However, also notice that the lu_solve
# takes in self.b, which is the RHS. So, do a for loop to do many different solving
# for the lu_solve, but the lu_factor will only need to be done once.
# TLDR: we will have a lot of different RHS due to the different Ieq that is needed,
# thus we would need to do many different lu_solve for each of the self.b

class LUSolver:
    def __init__(self, A, b):
        self.A = A
        self.b = b

    def lu_solve(self):
        lu, piv = lu_factor(self.A)
        x = lu_solve((lu, piv), self.b)

        return x

    def lu_factor_only(self, A):
        lu, piv = lu_factor(A)
        self.lu = lu
        self.piv = piv

    def lu_solve_only(self, B):
        x = lu_solve((self.lu, self.piv), B)
        return x

class Circuit:
    def __init__(self, i, j):
        self.i = i
        self.j = j
        self.unique_voltages = []
        self.unique_voltage_i = []
        self.unique_voltage_j = []

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
            self.lhs[i - 1][l - 1] += value
        if(i != 0 and l == 0):
            self.lhs[i - 1][k - 1] += -1*value
        if(j != 0 and k == 0):
            self.lhs[j - 1][l - 1] += -1*value
        if(j != 0 and l == 0):
            self.lhs[j - 1][k - 1] += value
        if(i != 0 and j != 0 and l != 0 and k != 0):
            self.lhs[i - 1][k - 1] += value
            self.lhs[i - 1][l - 1] += -1 * value
            self.lhs[j - 1][k - 1] += -1 * value
            self.lhs[j - 1][l - 1] += value

    """
    Dealing with the B and C parts of the LHS matrix
    """
    def current_control_voltage_update_lhs(self, value, i, j, volt, code):
        ncm = 0
        ncp = 0
        if(volt in self.unique_voltages):
            index_j = self.unique_voltages.index(volt) + self.i - 1
            ncp = self.unique_voltage_i[self.unique_voltages.index(volt)]
            ncm = self.unique_voltage_j[self.unique_voltages.index(volt)]
        else:
            return

        if(code in self.unique_voltages):
            index_k = self.unique_voltages.index(code) + self.i - 1
        else:
            self.unique_voltages.append(code)
            self.unique_voltage_i.append(i)
            self.unique_voltage_j.append(j)
            index_k = self.unique_voltages.index(code) + self.i - 1

        if(i == 0 and ncp == 0):
            self.lhs[j - 1][index_k] = -1
            self.lhs[ncm - 1][index_j] = -1
            self.lhs[index_k][j - 1] = -1
            self.lhs[index_j][ncm - 1] = -1
            self.lhs[index_k][index_j] = -1 * value
        if(j == 0 and ncp == 0):
            self.lhs[i - 1][index_k] = 1
            self.lhs[ncm - 1][index_j] = 1
            self.lhs[index_k][i - 1] = 1
            self.lhs[index_j][ncm - 1] = 1
            self.lhs[index_k][index_j] = -1 * value
        if(i == 0 and ncm == 0):
            self.lhs[j - 1][index_k] = -1
            self.lhs[ncp - 1][index_j] = 1
            self.lhs[index_k][j - 1] = -1
            self.lhs[index_j][ncp - 1] = 1
            self.lhs[index_k][index_j] = -1 * value
        if(j == 0 and ncp == 0):
            self.lhs[i - 1][index_k] = 1
            self.lhs[ncm - 1][index_j] = -1
            self.lhs[index_k][i - 1] = 1
            self.lhs[index_j][ncm - 1] = -1
            self.lhs[index_k][index_j] = -1 * value
        if(i != 0 and j != 0 and ncp != 0 and ncm != 0):
            self.lhs[i - 1][index_k] = 1
            self.lhs[ncp - 1][index_j] = 1
            self.lhs[j - 1][index_k] = -1
            self.lhs[ncm - 1][index_j] = -1
            self.lhs[index_k][i - 1] = 1
            self.lhs[index_j][ncp - 1] = 1
            self.lhs[index_k][j - 1] = -1
            self.lhs[index_j][ncm - 1] = -1
            self.lhs[index_k][index_j] = -1 * value
        
    def current_control_current_update_lhs(self, value, i, j, volt):
        ncm = 0
        ncp = 0
        if(volt in self.unique_voltages):
            index = self.unique_voltages.index(volt) + self.i - 1
            ncp = self.unique_voltage_i[self.unique_voltages.index(volt)]
            ncm = self.unique_voltage_j[self.unique_voltages.index(volt)]
        else:
            return

        if(i == 0 and ncp == 0):
            self.lhs[j - 1][index] = -1 * value
            self.lhs[ncm - 1][index] = -1
            self.lhs[index][ncm - 1] = -1
        if(j == 0 and ncp == 0):
            self.lhs[i - 1][index] = value
            self.lhs[ncm - 1][index] = 1
            self.lhs[index][ncm - 1] = -1
        if(i == 0 and ncm == 0):
            self.lhs[j - 1][index] = -1 * value
            self.lhs[ncp - 1][index] = 1
            self.lhs[index][ncp - 1] = 1
        if(j == 0 and ncp == 0):
            self.lhs[i - 1][index] = value
            self.lhs[ncm - 1][index] = -1
            self.lhs[index][ncm - 1] = 1
        if(i != 0 and j != 0 and ncp != 0 and ncm != 0):
            self.lhs[i - 1][index] = value
            self.lhs[j - 1][index] = -1 * value
            self.lhs[ncp - 1][index] = 1
            self.lhs[ncp - 1][index] = -1
            self.lhs[index][ncp - 1] = 1
            self.lhs[index][ncm - 1] = -1

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
            self.unique_voltage_i.append(i)
            self.unique_voltage_j.append(j)
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
        self.rhs[i - 1][0] = value

    def i_current_update_rhs(self, value, i, j):
        self.rhs[i - 1][0] = value
        self.rhs[j - 1][0] = -1*value
    
    def return_rhs(self):
        print(self.rhs)
        return self.rhs


if __name__ == "__main__":
    node_amount = int(sys.argv[1])
    num_volt = int(sys.argv[2])
    if(len(sys.argv) > 5):
        time_finish = float(sys.argv[4])
        time_resolution = int(sys.argv[5])
        time_delta = time_finish / time_resolution
    save_matrix_txt = False

    c = Circuit(node_amount + num_volt, node_amount + num_volt)
    c.create_lhs()
    c.create_rhs()

    resistors = []
    voltages = []
    currents = []
    vccs = []
    ccvs = []
    cccs = []
    cap = []
    ind = []

    capacitor_current_index = None
    capacitor_div_time = 0

    with open(str(sys.argv[3]), "r") as netlist_file:
        # Read the contents of the file into a string variable
        netlist = netlist_file.read()
        lines = netlist.split("\n")
        for line in lines:
            if line.strip() != "":
                parts = line.split()
                element = None
                node_values = None
                element_values = None
                volt_ref = None
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
                elif(len(parts) == 5):
                    element = parts[0]
                    node_values = parts[1:3]
                    volt_ref = parts[3]
                    element_value = parts[4]
                
                if element.startswith("R"):
                    resistors.append([element] + nodes_values + [element_value])
                elif element.startswith("V"):
                    voltages.append([element] + nodes_values + [element_value])
                elif element.startswith("I"):
                    currents.append([element] + nodes_values + [element_value])
                elif element.startswith("G"):
                    vccs.append([element] + nodes_values + [element_value])
                elif element.startswith("F"):
                    ccvs.append([element] + nodes_values + [volt_ref] + [element_value])
                elif element.startswith("H"):
                    cccs.append([element] + nodes_values + [volt_ref] + [element_value])
                elif element.startswith("C"):
                    cap.append([element] + nodes_values + [element_value])
                    capacitor_div_time = float(element_value)/time_delta
                    resistors.append([element] + nodes_values + [str(float(1/capacitor_div_time))])
                    currents.append([element] + nodes_values + [0])
                    capacitor_current_index = len(currents) - 1
                elif element.startswith(".text"):
                    save_matrix_txt = True

    for r in resistors:
        code = r[0]
        np = int(r[1])
        nm = int(r[2])
        value = float(r[3])

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
        value = float(i[3])

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
        value = float(g[5])

        c.voltage_control_current_update_lhs(value, np, nm, ncp, ncm)

    for f in ccvs:
        code = f[0]
        np = int(f[1])
        nm = int(f[2])
        volt = f[3]
        value = int(f[4])

        c.current_control_voltage_update_lhs(value, np, nm, volt, code)
    
    for h in cccs:
        code = h[0]
        np = int(h[1])
        nm = int(h[2])
        volt = h[3]
        value = int(h[4])

        c.current_control_current_update_lhs(value, np, nm, volt)
    
    if save_matrix_txt:
        f = open("test.out", 'w')
        sys.stdout = f
    print("=======LHS Matrix======")
    lhs = c.return_lhs()
    print("=======RHS Matrix======")
    rhs = c.return_rhs()

    print("=======LU Decomposition=======")
    l = LUSolver(lhs, rhs)
    l.lu_factor_only(lhs)

    cap_volt = []
    # For loop here
    if len(cap) > 0:
        for i in range(0, time_resolution):
            rhs = c.return_rhs()
            solve_out = l.lu_solve_only(rhs)
            print(solve_out)
            print("=======")
            new_volt = solve_out[1][0]
            cap_volt.append(new_volt)
            print(new_volt)
            currents[capacitor_current_index][3] = new_volt * float(capacitor_div_time)
            for i in currents:
                code = i[0]
                np = int(i[1])
                nm = int(i[2])
                value = float(i[3])

                if(np == 0):
                    c.i_current_update_rhs_zero(value, nm, nm)
                elif(nm == 0):
                    c.i_current_update_rhs_zero(value, np, np)
                else:
                    c.i_current_update_rhs(value, np, nm)
    
        plt.plot(cap_volt)
        plt.show()
    else:
        solve_out = l.lu_solve_only(rhs)
        print(solve_out)
        print("=======")
        print(solve_out[1][0])

    print(currents[0][3])
    if save_matrix_txt:
        f.close()