"""
Reference: http://www.pstcc.edu/facstaff/jwlamb/Math1630/6.4.pdf
Requires Python 3.10
"""
import numpy as np
from fractions import Fraction

DEBUG = False
M = Fraction(1e9)
tolerance = 1e-6
ZERO = Fraction(0)
ONE = Fraction(1)


def simplex(constraints, objective):
    """
    :param constraints: array of [...coefficients, constraint type, RHS]
    constraint types: -1: <=; 0: =; 1: >=
    :param objective: array of numbers, objective to maximise
    :return: optimised result
    """
    num_vars = len(objective)
    # prelim check: lengths are correct
    for arr in constraints:
        if len(arr) != num_vars + 2:
            raise ValueError(f'len({arr}) != {num_vars + 2}')
        if abs(arr[num_vars]) > 1:
            raise ValueError(f'invalid constraint type {arr[num_vars]}')
    # convert everything to fraction to prevent float errors
    constraints = [[Fraction(y) for y in x] for x in constraints]
    objective = [Fraction(x) for x in objective]
    # prelim pass: how many slack and artificial vars do we need?
    # for <= (-1) constraints: 1 slack var (+)
    # for = (0) constraints: 1 artificial var (+)
    # for >= (1) constraints: 1 slack var (-) and 1 artificial var (+)
    num_slack = 0
    num_artificial = 0
    num_constraints = len(constraints)
    inf = float("inf")
    for arr in constraints:
        side = arr[num_vars]
        rhs = arr[num_vars + 1]
        if rhs < 0:
            side = -side
        if side == -1:
            num_slack += 1
        elif side == 0:
            num_artificial += 1
        elif side == 1:
            num_slack += 1
            num_artificial += 1
    # construct the simplex
    if DEBUG:
        print(f'{num_slack} slacks {num_artificial} artificials')
    total_vars = num_vars + num_artificial + num_slack + 1  # +1: objective
    augmented = np.ndarray((num_constraints + 1, total_vars + 1), dtype=Fraction)  # objective and RHS
    augmented[:, :] = Fraction(0, 1)
    idx_slack = -1
    idx_artificial = -1
    idx_constraint = -1
    for arr in constraints:
        idx_constraint += 1
        side = arr[num_vars]
        rhs = arr[num_vars + 1]
        augmented[idx_constraint][0:num_vars] = arr[0:num_vars]
        augmented[idx_constraint][total_vars] = arr[num_vars + 1]
        if rhs < 0:
            side = -side
            augmented[idx_constraint] *= -1
        if side == -1:
            idx_slack += 1
            augmented[idx_constraint][num_vars + idx_slack] = ONE
        elif side == 0:
            idx_artificial += 1
            augmented[idx_constraint][num_vars + num_slack + idx_artificial] = ONE
        elif side == 1:
            idx_slack += 1
            idx_artificial += 1
            augmented[idx_constraint][num_vars + idx_slack] = -ONE
            augmented[idx_constraint][num_vars + num_slack + idx_artificial] = ONE
    # objective
    augmented[-1][0:num_vars] = objective[0:num_vars]
    augmented[-1] *= -1
    augmented[-1][total_vars - 1] = ONE
    # penalty for artificial vars
    for i in range(num_artificial):
        augmented[-1][-i - 3] = M  # big M
    if DEBUG:
        print(augmented)
    # eliminate M from the objective row
    for i in range(num_artificial):
        augmented[-1] -= M * augmented[-2 - i]
    basics = [0 for _ in constraints]
    basics.append(-1)  # the objective
    # choose initial basic variables (slack and artificial ones)
    basics[:-1] = range(num_vars, num_vars + num_constraints)
    if DEBUG:
        print(augmented)
        print(basics)
    num_iter = 0
    while True:
        num_iter += 1
        # optimality test: if all coefficients in row 0 are >= 0 then is optimal
        optimal = True
        for coef in range(0, total_vars - 1):
            if augmented[-1, coef] < 0:
                optimal = False
                break
        if optimal:
            break
        # choose entering basic variable: min in row -1
        max_abs = 0
        entering_idx = -1
        for coef in range(0, total_vars - 1):
            if augmented[-1, coef] < max_abs:
                entering_idx = coef
                max_abs = augmented[-1, coef]
        if DEBUG:
            print(f"iteration {num_iter}: entering basic variable is x{entering_idx}")
        # choose leaving basic variable: row with the least RHS/leaving, s.t. leaving>0
        with np.errstate(divide='ignore', invalid='ignore'):
            # coefs = np.where(
            #     augmented[:, entering_idx] > 0,
            #     augmented[:, total_vars] / augmented[:, entering_idx],
            #     inf)
            coefs = [x / y if y > 0 else inf for x, y in zip(augmented[:, total_vars], augmented[:, entering_idx])]
        leaving_idx = -1
        min_coef = inf
        idx = 0
        for coef in coefs:
            if coef < min_coef:
                leaving_idx = idx
                min_coef = coef
            idx += 1
        if DEBUG:
            print(f"iteration {num_iter}: leaving basic variable is x{basics[leaving_idx]} (row {leaving_idx})")
        # change pivot to 1
        augmented[leaving_idx] /= augmented[leaving_idx, entering_idx]
        # "matrix operation"
        for j in range(num_constraints + 1):
            if j == leaving_idx:
                continue
            augmented[j] -= augmented[leaving_idx] * augmented[j, entering_idx]
        basics[leaving_idx] = entering_idx
        if DEBUG:
            print(augmented)
            print(basics)
    # keep the basic variables (others are 0)
    result = np.matrix([augmented[:, total_vars - 1]])
    basics = np.sort(basics)
    for i in basics[1:]:
        result = np.concatenate((result, [augmented[:, i]]))
    result = np.asarray(result.T).tolist()
    rhs = augmented[:, total_vars]
    if DEBUG:
        print(result, rhs)
    # rearrange rows for the solution
    # x = np.linalg.solve(result, rhs)
    x = [ZERO for _ in basics]
    x.append(ZERO)
    for row, value in zip(result, rhs):
        try:
            idx = row.index(1)
            x[idx] = value
        except ValueError as e:
            if row[-1] != ZERO:
                # 0 != 0
                return None
    # x0, ..., xi, Z
    vec = [ZERO for _ in range(total_vars)]
    j = 1
    for i in basics[1:]:
        vec[i] = x[j]
        j += 1
    vec[total_vars - 1] = x[0]
    sol = vec[0:num_vars]
    sol.append(vec[-1])
    # check of the solution is correct
    for i in constraints:
        rhs = sum(x * y for x, y in zip(i[:-2], sol[:-1]))
        if not((abs(rhs - i[-1]) <= tolerance and i[-2] == 0) or (rhs <= i[-1] + tolerance and i[-2] == -1) or (rhs >= i[-1] - tolerance and i[-2] == 1)):
            if DEBUG:
                print(f'result incorrect, constraint: {i}; sol: {sol}; rhs: {rhs}; value: {i[-1]}')
            return None
    return sol


def mip(constraints, additional_constraints, objective):
    """
    :param constraints: array of [...coefficients, constraint type, RHS]
    :param additional_constraints: array of chars, I: integer; B: binary; real otherwise
    :param objective: array of numbers, objective to maximise
    :return: optimised result
    """
    inf = float("inf")
    best_z = -inf
    best_sol = None
    constraints = constraints.copy()
    og_constraints = constraints.copy()
    objective = [Fraction(x) for x in objective]
    sol = simplex(constraints, objective)
    num_vars = len(objective)
    cached_sols = {(): sol}
    curr_bounds = ()
    bfs = []
    if sol is None:
        # LP relaxation has no solution
        return None
    while True:
        # just bfs it (tm)
        if DEBUG:
            print(f'considering {sol}')
        violation = -1
        for i in range(num_vars):
            if sol[i].denominator > 1 and (additional_constraints[i] == 'I' or additional_constraints[i] == 'B'):
                violation = i
                break
        z = sum(x * y for x, y in zip(objective, sol[:-1]))
        if violation < 0:
            # no violation
            if z > best_z:
                best_z = z
                best_sol = sol
        else:
            if z < best_z:
                # fanthomed: will never beat the current best
                continue
            if additional_constraints[violation] == 'I':
                # split into <= floor(x) and >= ceil(x)
                ratio = sol[violation].as_integer_ratio()
                floored = Fraction(ratio[0] // ratio[1])
                c1 = [ZERO for _ in constraints[0]]
                c1[violation] = ONE
                c1[-2] = -ONE
                c1[-1] = floored
                c2 = [ZERO for _ in constraints[0]]
                c2[violation] = ONE
                c2[-2] = ONE
                c2[-1] = floored + 1
                b1 = list(curr_bounds)
                b2 = list(curr_bounds)
                b1.append(tuple(c1))
                b2.append(tuple(c2))
                bfs.append(tuple(b1))
                bfs.append(tuple(b2))
            elif additional_constraints[violation] == 'B':
                # split into =0 and =1
                c1 = [ZERO for _ in constraints[0]]
                c1[violation] = ONE
                c1[-2] = ZERO
                c1[-1] = ZERO
                c2 = [ZERO for _ in constraints[0]]
                c2[violation] = ONE
                c2[-2] = ZERO
                c2[-1] = ONE
                b1 = list(curr_bounds)
                b2 = list(curr_bounds)
                b1.append(tuple(c1))
                b2.append(tuple(c2))
                bfs.append(tuple(b1))
                bfs.append(tuple(b2))
        s = None
        while s is None:
            # this loop fanthoms infeasible ones
            if len(bfs) > 0:
                curr_bounds = bfs.pop(0)
                constraints = og_constraints.copy()
                if curr_bounds in cached_sols:
                    # optimisation: avoid recalculations
                    sol = cached_sols[curr_bounds]
                else:
                    for i in curr_bounds:
                        constraints.append(list(i))
                    if DEBUG:
                        print(f'sol: {sol}')
                        print('extra constraints')
                        for j in curr_bounds:
                            print(f'  {j.index(1)} {j[-2]} {j[-1]}')
                    s = simplex(constraints, objective)
                    if s is not None:
                        sol = s
                        cached_sols[curr_bounds] = s
            else:
                return best_sol


def main():
    np.set_printoptions(linewidth=1000, suppress=True)
    # lp1()
    # lp2()
    print(mip([
        [2, 4, 5, 0, 0, 0, -1, 100],
        [1, 1, 1, 0, 0, 0, -1, 30],
        [10, 5, 2, 0, 0, 0, -1, 204],
        [1, 0, 0, -M, 0, 0, -1, 0],
        [0, 1, 0, 0, -M, 0, -1, 0],
        [0, 0, 1, 0, 0, -M, -1, 0],
    ], ['I', 'I', 'I', 'B', 'B', 'B'], [52, 30, 20, -500, -400, -300]))


if __name__ == "__main__":
    main()
