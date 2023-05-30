import numpy as np
from scipy.linalg import lu_factor, lu_solve
import sys
import matplotlib.pyplot as plt

import time

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
    
    print("module lim_sv (input logic clk, input logic reset, output logic done);")
    start_time = time.time()
    time_res = 100000
    time_res_bit_length = time_res.bit_length()

    branches_list, nodes_list = parse_text_file('test_file_5.txt')
    b_len = len(branches_list).bit_length()
    n_len = len(nodes_list).bit_length()
    print("\n\t// Declaring variables")
    print(f"\tlogic [{time_res_bit_length-1}:0] i;")
    print(f"\tparameter T = {time_res};")
    print(f"\tparameter B = {len(branches_list)};")
    print(f"\tparameter N = {len(nodes_list)};")
    print(f"\tlogic [{b_len-1}:0] b;")
    print(f"\tlogic [{n_len-1}:0] n;")
    print(f"\treal C;")
    print(f"\treal G;")
    print(f"\treal H;")
    print(f"\treal L;")
    print(f"\treal R;")
    print(f"\treal t;")

    print("\n\t// Declaring RAM")
    print(f"\treal currents [{b_len-1}:0];")
    print(f"\treal voltages [{n_len-1}:0];")
    print(f"\treal prev_currents [{b_len-1}:0];")
    print(f"\treal prev_voltages [{n_len-1}:0];")
    currents = np.zeros((len(branches_list), time_res+1))
    voltages = np.zeros((len(nodes_list), time_res+1))

    print(f"\n\t// Declaring state logic")
    print(f"\tlogic [2:0] state;")

    print(f"\n\t//State Machine Loop in SV")
    print(f"\talways @(posedge clk or posedge reset) begin\n"\
    "\t\tif (reset) begin\n"\
    "\t\t\t// Reset the state and counters\n"\
    "\t\t\ti <= 0;\n"\
    "\t\t\tb <= 0;\n"\
    "\t\t\tn <= 0;\n"\
    "\t\t\tdone <= 0;\n"\
    "\t\t\tt <= T;\n"\
    "\t\t\tstate <= 3'b000;\n"\
    "\t\tend else begin")
    print(f"\t\t\tcase(state)")

    # ------------------------------------------
    print(f"\t\t\t\t// Outer loop")
    print(f"\t\t\t\t3'b000:")
    print(f"\t\t\t\tif (i < T) begin")
    print(f"\t\t\t\t\ti <= i + 1;")
    print(f"\t\t\t\t\tstate <= 3'b001;")
    print(f"\t\t\t\t\tif (i == 0) begin")
    for n in range(0, len(nodes_list)):
        print(f"\t\t\t\t\t\tprev_voltages[{n}] <= 0;")
    for b in range(0, len(branches_list)):
        print(f"\t\t\t\t\t\tprev_currents[{b}] <= 0;")
    print(f"\t\t\t\t\tend")
    print(f"\t\t\t\t\telse begin")
    for n in range(0, len(nodes_list)):
        print(f"\t\t\t\t\t\tprev_voltages[{n}] <= voltages[{n}];")
    for b in range(0, len(branches_list)):
        print(f"\t\t\t\t\t\tprev_currents[{b}] <= currents[{b}];")
    print(f"\t\t\t\t\tend")
    print(f"\t\t\t\tend")
    print(f"\t\t\t\telse begin")
    print(f"\t\t\t\t\t$display(\"Done\");")
    print(f"\t\t\t\t\tdone <= 1;")
    print(f"\t\t\t\t\tstate <= 3'b101;")
    print(f"\t\t\t\tend")

    # ------------------------------------------
    print(f"\n\t\t\t\t//Load for First inner loop")
    print(f"\t\t\t\t3'b001:")
    print(f"\t\t\t\tif (n < N) begin")
    for n in range(0, len(nodes_list)):
        print(f"\t\t\t\t\tif(n == {n}) begin")
        C = find_C_value(nodes_list[n])
        G = find_G_value(nodes_list[n])
        H = find_H_value(nodes_list[n])
        print(f"\t\t\t\t\t\tC <= {C};")
        print(f"\t\t\t\t\t\tG <= {G};")
        print(f"\t\t\t\t\t\tH <= {H};")
        print(f"\t\t\t\t\tend")
    print(f"\t\t\t\t\tn <= n + 1;")
    print(f"\t\t\t\t\tstate <= 3'b010;")
    print(f"\t\t\t\tend else begin")
    print(f"\t\t\t\t\tn <= 0;")
    print(f"\t\t\t\t\tstate <= 3'b011;")
    print(f"\t\t\t\tend")

    # ------------------------------------------
    print(f"\n\t\t\t\t//First inner loop")
    print(f"\t\t\t\t3'b010: begin")
    print(f"\t\t\t\tvoltages[n-1] <= (((C*prev_voltages[n-1])/(1/t)) + H - prev_currents[0]) / ((C/(1/t)) + G);")
    print(f"\t\t\t\tstate <= 3'b001;")
    print(f"\t\t\t\tend")

    #  ------------------------------------------
    print(f"\n\t\t\t\t//Load for Second inner loop")
    print(f"\t\t\t\t3'b011:")
    print(f"\t\t\t\tif (b < B) begin")
    for b in range(0, len(branches_list)):
        print(f"\t\t\t\t\tif(b == {b}) begin")
        L = find_L_value(branches_list[b])
        R = find_R_value(branches_list[b])
        print(f"\t\t\t\t\t\tL <= {L};")
        print(f"\t\t\t\t\t\tR <= {R};")
        print(f"\t\t\t\t\tend")
    print(f"\t\t\t\t\tb <= b + 1;")
    print(f"\t\t\t\t\tstate <= 3'b100;")
    print(f"\t\t\t\tend else begin")
    print(f"\t\t\t\t\tb <= 0;")
    print(f"\t\t\t\t\tstate <= 3'b000;")
    print(f"\t\t\t\tend")

    # ------------------------------------------
    print(f"\n\t\t\t\t//Second inner loop")
    print(f"\t\t\t\t3'b100: begin")
    print(f"\t\t\t\tcurrents[b-1] <= prev_currents[b-1] + (1/t)/L * (voltages[b-1] - voltages[b] - R*prev_currents[b-1]);")
    print(f"\t\t\t\tstate <= 3'b011;")
    print(f"\t\t\t\tend")

    # ------------------------------------------
    print(f"\n\t\t\t\t//Second inner loop")
    print(f"\t\t\t\t3'b101:")
    print(f"\t\t\t\tstate <= 3'b101;")

    print(f"\t\t\tendcase")
    print(f"\t\tend")
    print(f"\tend")

    # for i in range(0, time_res):
    #     # Node Voltages
    #     for n in range(0, len(nodes_list)):
    #         C = find_C_value(nodes_list[n])
    #         G = find_G_value(nodes_list[n])
    #         H = find_H_value(nodes_list[n])
    #         voltages[n, i+1] = ((C*voltages[n, i]/(1/time_res)) + H - currents[0, i]) / ((C/(1/time_res)) + G)

    #     # Branch Currents
    #     for b in range(0, len(branches_list)):
    #         L = find_L_value(branches_list[b])
    #         R = find_R_value(branches_list[b])
    #         currents[b, i+1] = currents[b, i] + (1/time_res)/L * (voltages[b, i+1] - voltages[b+1, i+1] - R*currents[b, i])


    print("\nendmodule")

    # print("--- %s seconds ---" % (time.time() - start_time))
    # plt.plot(currents[0, :])
    # plt.title("Graph of Current")
    # plt.xlabel("Time Step")
    # plt.ylabel("Current (A)")
    # plt.show()
    # plt.plot(voltages[0, :])
    # plt.title("Graph of Voltage at Node i")
    # plt.xlabel("Time Step")
    # plt.ylabel("Voltage (V)")
    # plt.show()
    # plt.plot(voltages[1, :])
    # plt.title("Graph of Voltage at Node j")
    # plt.xlabel("Time Step")
    # plt.ylabel("Voltage (V)")
    # plt.show()
