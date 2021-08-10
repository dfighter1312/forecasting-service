from arch.arch import ArchitectureGen
from models.linear_reg import LinearRegression

if __name__ == "__main__":
    # Testing up to Architecture Generation part for Simple Linear Regression model
    __C = 2
    ArchitectureGen().arch_gen(LinearRegression(__C))