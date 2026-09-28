"""core@v1 — 60 exact-answer probes in three difficulty tiers.

Static, deterministic, and unambiguous by design: every answer is a single
number, so extraction and comparison are exact. Tiers give the accuracy-vs-
effort metric dynamic range (low effort should still clear easy items but
may miss hard ones — that gradient is what calibration reads).
"""

from . import Probe

PROBES: list[Probe] = [
    # ---- easy (20): arithmetic fluency ----
    Probe("e01", "easy", "exact", "What is 17 + 26?", "43"),
    Probe("e02", "easy", "exact", "What is 8 * 7 + 6?", "62"),
    Probe("e03", "easy", "exact", "What is 144 / 12?", "12"),
    Probe("e04", "easy", "exact", "What is 15% of 80?", "12"),
    Probe("e05", "easy", "exact", "A train travels 60 km/h for 2.5 hours. How many km does it cover?", "150"),
    Probe("e06", "easy", "exact", "What is 3.5 + 2.25?", "5.75"),
    Probe("e07", "easy", "exact", "What is (18 - 7) * 3?", "33"),
    Probe("e08", "easy", "exact", "What is 2 to the power of 10?", "1024"),
    Probe("e09", "easy", "exact", "What is the sum of all integers from 1 to 20?", "210"),
    Probe("e10", "easy", "exact", "What is 45 * 11?", "495"),
    Probe("e11", "easy", "exact", "If 3 pens cost 12 dollars, how much do 7 pens cost?", "28"),
    Probe("e12", "easy", "exact", "What is 100 - 37?", "63"),
    Probe("e13", "easy", "exact", "What is half of 2 to the power of 7?", "64"),
    Probe("e14", "easy", "exact", "What is 12 * 12?", "144"),
    Probe("e15", "easy", "exact", "What is 7 factorial divided by 6 factorial?", "7"),
    Probe("e16", "easy", "exact", "What is the average of 4, 9 and 11?", "8"),
    Probe("e17", "easy", "exact", "How many quarters are in 3 dollars?", "12"),
    Probe("e18", "easy", "exact", "What is 40% of 250?", "100"),
    Probe("e19", "easy", "exact", "What is 9*9 + 9*1?", "90"),
    Probe("e20", "easy", "exact", "What is 5 to the power of 3?", "125"),
    # ---- medium (20): multi-step ----
    Probe("m01", "medium", "exact", "What is 23 * 47?", "1081"),
    Probe("m02", "medium", "exact", "What is 2 to the power of 15?", "32768"),
    Probe("m03", "medium", "exact", "What is the sum of all integers from 1 to 100?", "5050"),
    Probe("m04", "medium", "exact", "If 5 machines make 5 widgets in 5 minutes, how many minutes do 100 machines need to make 100 widgets?", "5"),
    Probe("m05", "medium", "exact", "What is the least common multiple of 12 and 18?", "36"),
    Probe("m06", "medium", "exact", "What is the greatest common divisor of 84 and 132?", "12"),
    Probe("m07", "medium", "exact", "Simple interest on 2000 dollars at 5% per year for 3 years, in dollars?", "300"),
    Probe("m08", "medium", "exact", "A 240 dollar item is discounted 15%. What is the final price in dollars?", "204"),
    Probe("m09", "medium", "exact", "A car covers 90 km in 1 hour 15 minutes. What is its average speed in km/h?", "72"),
    Probe("m10", "medium", "exact", "1000 dollars grows 10% per year, compounded. Value after 2 years, in dollars?", "1210"),
    Probe("m11", "medium", "exact", "Solve for x: 3x + 7 = 31", "8"),
    Probe("m12", "medium", "exact", "Solve for x: x/4 + 3 = 10", "28"),
    Probe("m13", "medium", "exact", "Area of a circle with radius 7, using pi = 22/7?", "154"),
    Probe("m14", "medium", "exact", "A right triangle has legs 5 and 12. What is the hypotenuse?", "13"),
    Probe("m15", "medium", "exact", "Two angles of a triangle are 40 and 65 degrees. What is the third?", "75"),
    Probe("m16", "medium", "exact", "7 workers finish a job in 12 days. How many days do 14 workers need?", "6"),
    Probe("m17", "medium", "exact", "A square has area 144. What is its perimeter?", "48"),
    Probe("m18", "medium", "exact", "Three consecutive integers sum to 51. What is the largest?", "18"),
    Probe("m19", "medium", "exact", "Probability of rolling an even number on a fair die, as a decimal?", "0.5"),
    Probe("m20", "medium", "exact", "What is 3 to the power of 5 minus 3 to the power of 3?", "216"),
    # ---- hard (20): reasoning ----
    Probe("h01", "hard", "exact", "What is the square root of 1764?", "42"),
    Probe("h02", "hard", "exact", "What is 2 to the power of 20?", "1048576"),
    Probe("h03", "hard", "exact", "What is the digit sum of 343?", "10"),
    Probe("h04", "hard", "exact", "How many prime numbers are less than 20?", "8"),
    Probe("h05", "hard", "exact", "What is the sum of the first 10 odd numbers?", "100"),
    Probe("h06", "hard", "exact", "How many ways can you choose 3 items from 8, order irrelevant?", "56"),
    Probe("h07", "hard", "exact", "How many ordered arrangements of 2 items from 5?", "20"),
    Probe("h08", "hard", "exact", "Solve for x: 2 to the power of x equals 1024.", "10"),
    Probe("h09", "hard", "exact", "2 to what power equals 4096?", "12"),
    Probe("h10", "hard", "exact", "x squared is 169. What is the positive x?", "13"),
    Probe("h11", "hard", "exact", "If f(n) = 2n + 3, what is f(f(2))?", "17"),
    Probe("h12", "hard", "exact", "A price rises from 80 to 100. What is the percent increase?", "25"),
    Probe("h13", "hard", "exact", "The average of 5 consecutive integers is 30. What is the middle one?", "30"),
    Probe("h14", "hard", "exact", "What is 12 factorial divided by 10 factorial?", "132"),
    Probe("h15", "hard", "exact", "How many positive factors does 60 have?", "12"),
    Probe("h16", "hard", "exact", "What is the sum of interior angles of a regular hexagon, in degrees?", "720"),
    Probe("h17", "hard", "exact", "What is the remainder when 7 to the power of 100 is divided by 5?", "1"),
    Probe("h18", "hard", "exact", "What is the smallest positive integer divisible by every integer from 1 to 6?", "60"),
    Probe("h19", "hard", "exact", "A fair coin is flipped 3 times. Probability of exactly 2 heads, as a decimal?", "0.375"),
    Probe("h20", "hard", "exact", "What is the sum of the squares of the first 5 positive integers?", "55"),
]
