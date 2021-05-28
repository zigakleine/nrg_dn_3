import numpy as np
import os

# source: https://en.wikipedia.org/wiki/Centripetal_Catmull%E2%80%93Rom_spline
def catmull_rom_spline(P0, P1, P2, P3, nPoints):
    """
    P0, P1, P2, and P3 should be (x,y) point pairs that define the Catmull-Rom spline.
    nPoints is the number of points to include in this curve segment.
    """
    # Convert the points to numpy so that we can do array multiplication
    P0, P1, P2, P3 = map(np.array, [P0, P1, P2, P3])

    # Parametric constant: 0.5 for the centripetal spline, 0.0 for the uniform spline, 1.0 for the chordal spline.
    alpha = 0
    # Premultiplied power constant for the following tj() function.
    alpha = alpha/2
    def tj(ti, Pi, Pj):
        xi, yi, zi = Pi
        xj, yj, zj = Pj
        return ((xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2)**alpha + ti

    # Calculate t0 to t4
    t0 = 0
    t1 = tj(t0, P0, P1)
    t2 = tj(t1, P1, P2)
    t3 = tj(t2, P2, P3)

    # Only calculate points between P1 and P2
    t = np.linspace(t1, t2, nPoints)

    # Reshape so that we can multiply by the points P0 to P3
    # and get a point for each value of t.
    t = t.reshape(len(t), 1)
    # print(t)
    A1 = np.nan_to_num((t1-t)/(t1-t0)*P0) + np.nan_to_num((t-t0)/(t1-t0)*P1)
    A2 = np.nan_to_num((t2-t)/(t2-t1)*P1) + np.nan_to_num((t-t1)/(t2-t1)*P2)
    A3 = np.nan_to_num((t3-t)/(t3-t2)*P2) + np.nan_to_num((t-t2)/(t3-t2)*P3)
    # print(A1)
    # print(A2)
    # print(A3)
    B1 = np.nan_to_num((t2-t)/(t2-t0)*A1) + np.nan_to_num((t-t0)/(t2-t0)*A2)
    B2 = np.nan_to_num((t3-t)/(t3-t1)*A2) + np.nan_to_num((t-t1)/(t3-t1)*A3)

    C = np.nan_to_num((t2-t)/(t2-t1)*B1) + np.nan_to_num((t-t1)/(t2-t1)*B2)
    return C




if __name__ == "__main__":

    inputs_ = ["input_00.txt", "input_01.txt", "input_02.txt", "input_03.txt"]
    fpss = [24, 24, 60, 24]


    for input_num, input in enumerate(inputs_):

        f_instr = open("./data/" + input, "r")
        instructions = []

        for instr_line in f_instr:
            instr_line_split = instr_line.split()
            filename = instr_line_split[0]
            miliseconds = instr_line_split[1]

            f_data = open("./data/" + filename, "r")
            file_data = f_data.readlines()
            vertex_indexes = []
            vertices = []

            for idx, file_line in enumerate(file_data):
                split_file_line = file_line.split()
                if len(split_file_line) > 0:
                    if split_file_line[0] == "v":
                        vertex_indexes.append(idx)
                        #print(split_file_line)
                        vertices.append([float(split_file_line[1]), float(split_file_line[2]), float(split_file_line[3])])

            # print(len(vertex_indices))
            f_data.close()
            # vertices = np.array(vertices)
            # vertex_indexes = np.array(vertex_indexes)

            instructions.append({"filename": filename, "ms": int(miliseconds), "file_data": file_data,
                                 "vertex_indexes": vertex_indexes, "vertices": vertices})

        f_instr.close()
        num_of_instr = len(instructions)

        outpoints = []


        current_frame_num = 0

        for i in range(num_of_instr - 1):

            instr_1 = instructions[i - 1]
            instr_2 = instructions[i]
            instr_3 = instructions[(i + 1) % num_of_instr]
            instr_4 = instructions[(i + 2) % num_of_instr]

            num_of_frames = int((fpss[input_num] * (instr_3["ms"] - instr_2["ms"]))/1000)
            # print(num_of_frames)
            # print(instr_1)
            # print(instr_2)
            # print(instr_3)
            # print(instr_4)
            # print("")

            for j in range(num_of_frames):
                outpoints.append([])

            for p1, p2, p3, p4 in zip(instr_1["vertices"], instr_2["vertices"], instr_3["vertices"], instr_4["vertices"]):
                # print(p1, p2, p3, p4)

                points = catmull_rom_spline(p1, p2, p3, p4, (num_of_frames + 1))
                points = points[:-1]

                for k in range(num_of_frames):
                    outpoints[k + current_frame_num].append(points[k])

            current_frame_num += num_of_frames

        outpoints = np.array(outpoints)
        # print(len(outpoints))

        outputfolder = "outputs0" + str(input_num)

        outputdir = os.path.join(os.getcwd(), outputfolder)

        try:
            os.mkdir(outputdir)
        except OSError as error:
            print(error)

        for idx, frame in enumerate(outpoints):
            vertex_indexes = instructions[0]["vertex_indexes"]
            file = instructions[0]["file_data"]

            # print(len(frame))

            for vertex_idx, vertex_index in enumerate(vertex_indexes):
                file[vertex_index] = "v " + str(frame[vertex_idx][0]) + " " + str(frame[vertex_idx][1]) + " " + \
                                     str(frame[vertex_idx][2]) + "\n"


            with open("./" + outputfolder + "/anim_0" + str(input_num) + "_" + str(idx + 1) + ".obj", 'w') as f:
                f.writelines(file)
