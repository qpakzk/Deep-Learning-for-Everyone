import numpy as np


nums = [n for n in range(5)]

print(nums) # [0, 1, 2, 3, 4]
print(nums[2:4]) # [2, 3]
print(nums[2:]) # [2, 3, 4]
print(nums[:2]) # [0, 1]
print(nums[:]) # [1, 2, 3, 4]
print(nums[:-1]) # [0, 1, 2, 3]
nums[2:4] = [8, 9]
print(nums) # [0, 1, 8, 9, 4]