import numpy as np
from scipy.linalg import lu_factor, lu_solve
import sys
import matplotlib.pyplot as plt

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
    def __init__(self, i, j, num_volt):
        self.i = i
        self.j = j
        self.num_volt = num_volt
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
            index = self.unique_voltages.index(code) + self.i - self.num_volt
        else:
            self.unique_voltages.append(code)
            self.unique_voltage_i.append(i)
            self.unique_voltage_j.append(j)
            index = self.unique_voltages.index(code) + self.i - self.num_volt
    
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

def parse_text_file(file_path):
    branches = []
    nodes = []
    current_list = []

    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()

            if line.startswith('.branch'):
                current_list = []
                branches.append(current_list)
            elif line.startswith('.node'):
                current_list = []
                nodes.append(current_list)
            else:
                current_list.append(line.split())

    return branches, nodes

def find_L_value(my_list):
    for sublist in my_list:
        if sublist[0][0] == 'L':
            return float(sublist[3])

def find_R_value(my_list):
    for sublist in my_list:
        if sublist[0][0] == 'R':
            return float(sublist[3])

def find_C_value(my_list):
    for sublist in my_list:
        if sublist[0][0] == 'C':
            return float(sublist[3])

def find_G_value(my_list):
    for sublist in my_list:
        if sublist[0][0] == 'R':
            return 1/float(sublist[3])

def find_H_value(my_list):
    for sublist in my_list:
        if sublist[0][0] == 'I':
            return float(sublist[3])
    return 0

if __name__ == "__main__":
    
    time = 1000

    branches_list, nodes_list = parse_text_file('test_file_5.txt')

    print("Branches:")
    for branch in branches_list[0]:
        print(branch)

    print(len(branches_list))
    print("\nNodes:")
    for node in nodes_list:
        for line in node:
            print(line)
        print()  # Add an empty line between nodes
    print(len(nodes_list))

    currents = np.zeros((len(branches_list), time+1))
    voltages = np.zeros((len(nodes_list), time+1))

    for i in range(0, time):
        # print(i)
        for n in range(0, len(nodes_list)):
            C = find_C_value(nodes_list[n])
            G = find_G_value(nodes_list[n])
            H = find_H_value(nodes_list[n])
            voltages[n, i+1] = ((C*voltages[n, i]/(1/time)) + H - currents[0, i]) / ((C/(1/time)) + G)

        for b in range(0, len(branches_list)):
            L = find_L_value(branches_list[b])
            R = find_R_value(branches_list[b])
            currents[b, i+1] = currents[b, i] + (1/time)/L * (voltages[b, i+1] - voltages[b+1, i+1] - R*currents[b, i])

    plt.plot(currents[0, :])
    plt.show()
    plt.plot(voltages[0, :])
    plt.show()
    plt.plot(voltages[1, :])
    plt.show()
