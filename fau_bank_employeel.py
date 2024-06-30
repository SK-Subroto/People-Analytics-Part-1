import pandas as pd
from pulp import *
from io import StringIO

def optimize_bank_teller_staffing(csv_file):
    # Read the CSV data
    df = pd.read_csv(csv_file, index_col=0)

    # Service level (customers per hour per employee)
    service_level = 8

    # Calculate the number of employees needed for each hour
    df['Employees_Needed'] = (df['Avg_Customer_Number'] / service_level).apply(lambda x: -(-x // 1))  # Ceiling division

    # Decision variables: number of workers for each shift
    shift1_employees = LpVariable("Shift1_Employees", lowBound=0, cat='Integer')
    shift2_employees = LpVariable("Shift2_Employees", lowBound=0, cat='Integer')

    # Create the problem
    prob = LpProblem("Bank_Teller_Staffing", LpMinimize)

    # Objective function: Minimize the total number of employees
    prob += shift1_employees + shift2_employees, "Total Employees"

    # Constraints: The number of employees must be sufficient for each hour
    for index, row in df.iterrows():
        if row['Shift 1'] == 'X':
            prob += shift1_employees >= row['Employees_Needed'], f"Shift1_{index}"
        if row['Shift 2'] == 'X':
            prob += shift2_employees >= row['Employees_Needed'], f"Shift2_{index}"

    # Solve the problem
    prob.solve()

    # Print the results
    print("Status:", LpStatus[prob.status])
    print("Objective value:", value(prob.objective))

    # Number of employees needed for each shift
    print(f"Shift 1: Employees Needed = {int(shift1_employees.value())}")
    print(f"Shift 2: Employees Needed = {int(shift2_employees.value())}")

    # Print detailed solver information
    print(f"Total time (CPU seconds): {prob.solutionCpuTime:.2f} (Wallclock seconds): {prob.solutionCpuTime:.2f}")

if __name__ == "__main__":
    csv_file = "data/fau_bank_shifts.csv"
    optimize_bank_teller_staffing(csv_file)
