import os
import pickle

def pickle_wrap(filename, callback, easy_override=False):
    if os.path.isfile(filename) and not easy_override:
        with open(filename, "rb") as file:
            return pickle.load(file)
    else:
        output = callback()
        with open(filename, "wb") as new_file:
            pickle.dump(output, new_file)
        return output

def get_date_mapping(starting_date_number):
    # doesn't work. issue where it gives dates like october 0th rather than sept 30th. easy to fix but also
    # doesn't account for some months being 31 days while others are 30
    first_num = int(str(starting_date_number)[-1])
    all_numbers = []
    for wk in range(1, 16):
        all_numbers.append((wk - 1)  * 7 + first_num)
        all_numbers.append((wk - 1)  * 7 + 3 + first_num)
        all_numbers.append((wk - 1)  * 7 + 4 + first_num)

    all_date_numbers = {}
    for num in all_numbers:
        day = num % 30
        month = 9 + int(num / 30)
        if day < 10:
            date_num = str(month) + '0' + str(day)
        else:
            date_num = str(month) + str(day)
        week = int((num - first_num) / 7) + 1
        all_date_numbers[date_num] = week
    return all_date_numbers
