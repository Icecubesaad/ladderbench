"""hard@v1 — 40 multi-step probes (the difficulty extension).

Written to widen the dial's dynamic range: the core set's thinking-token
magnitudes (34-65) under-stress hybrid models on general Q&A. These require
multi-step computation (number theory, combinatorics, sequences, word
problems) — the kind of task where harder effort settings should visibly
change trace length. Every answer is a single exact number, hand-verified.
"""

from . import Probe

PROBES: list[Probe] = [
    Probe("hh01", "hard+", "exact", "What is the sum of all prime numbers less than 50?", "328"),
    Probe("hh02", "hard+", "exact", "How many distinct arrangements are there of the letters in the word LEVEL?", "30"),
    Probe("hh03", "hard+", "exact", "What is (2 to the power 16 minus 2 to the power 10) divided by 2 to the power 8?", "252"),
    Probe("hh04", "hard+", "exact", "A fair coin is flipped 4 times. What is the probability of at least one head, as a decimal?", "0.9375"),
    Probe("hh05", "hard+", "exact", "What is the sum of the first 20 positive even numbers?", "420"),
    Probe("hh06", "hard+", "exact", "What is the least common multiple of 8, 12 and 20?", "120"),
    Probe("hh07", "hard+", "exact", "What is the greatest common divisor of 1071 and 462?", "21"),
    Probe("hh08", "hard+", "exact", "What is the units digit of 7 to the power 2026?", "9"),
    Probe("hh09", "hard+", "exact", "A triangle has sides 13, 14 and 15. What is its area?", "84"),
    Probe("hh10", "hard+", "exact", "What is the smallest positive integer x with 3x congruent to 12 modulo 15?", "4"),
    Probe("hh11", "hard+", "exact", "What is 100 written in binary, interpreted as a decimal number? (e.g. binary 110 is 110)", "1100100"),
    Probe("hh12", "hard+", "exact", "The binary number 101011 equals which decimal number?", "43"),
    Probe("hh13", "hard+", "exact", "3 to what power equals 729?", "6"),
    Probe("hh14", "hard+", "exact", "A number cubed equals 0.064. What is the number?", "0.4"),
    Probe("hh15", "hard+", "exact", "What is the sum of the interior angles of a 12-sided polygon, in degrees?", "1800"),
    Probe("hh16", "hard+", "exact", "How many diagonals does a regular octagon have?", "20"),
    Probe("hh17", "hard+", "exact", "Two fair dice are rolled. In how many outcomes is the first die strictly greater than the second?", "15"),
    Probe("hh18", "hard+", "exact", "5000 dollars grows 8% per year compounded. What is the value after 2 years, in dollars?", "5832"),
    Probe("hh19", "hard+", "exact", "What is the sum of the arithmetic series 3 + 7 + 11 + ... + 399?", "20100"),
    Probe("hh20", "hard+", "exact", "A geometric series starts 2, 6, 18, ... What is the sum of the first 8 terms?", "6560"),
    Probe("hh21", "hard+", "exact", "Worker A finishes a job in 6 hours, worker B in 4 hours. Working together, how many hours do they need?", "2.4"),
    Probe("hh22", "hard+", "exact", "A 120-meter train passes a pole in 8 seconds. What is its speed in meters per second?", "15"),
    Probe("hh23", "hard+", "exact", "What is the remainder when 2 to the power 100 is divided by 7?", "2"),
    Probe("hh24", "hard+", "exact", "What is the average of the first 50 positive odd numbers?", "50"),
    Probe("hh25", "hard+", "exact", "What is 1 cubed plus 2 cubed plus 3 cubed plus 4 cubed plus 5 cubed?", "225"),
    Probe("hh26", "hard+", "exact", "How many trailing zeros does 100 factorial have?", "24"),
    Probe("hh27", "hard+", "exact", "A rectangle has area 120 and diagonal 17. What is its perimeter?", "46"),
    Probe("hh28", "hard+", "exact", "A regular hexagon has side length 6. What is its perimeter?", "36"),
    Probe("hh29", "hard+", "exact", "If x plus 1/x equals 5, what is x squared plus 1 over x squared?", "23"),
    Probe("hh30", "hard+", "exact", "A number is increased by 20%, then the result is decreased by 20%. The final value is 96. What was the original number?", "100"),
    Probe("hh31", "hard+", "exact", "What is the angle between the hour and minute hands of a clock at 3:40, in degrees?", "130"),
    Probe("hh32", "hard+", "exact", "A father is 3 times as old as his son. In 12 years he will be twice as old. What is the sum of their current ages?", "48"),
    Probe("hh33", "hard+", "exact", "A two-digit number has digits summing to 9. Reversing the digits gives a number 27 less. What is the number?", "63"),
    Probe("hh34", "hard+", "exact", "A cube has surface area 96. What is its volume?", "64"),
    Probe("hh35", "hard+", "exact", "What is the sum of the infinite series 1 + 1/3 + 1/9 + 1/27 + ... ?", "1.5"),
    Probe("hh36", "hard+", "exact", "A committee of 4 is chosen from 5 men and 4 women, with exactly 2 women. How many committees are possible?", "60"),
    Probe("hh37", "hard+", "exact", "How many squares of any size are on an 8 by 8 chessboard?", "204"),
    Probe("hh38", "hard+", "exact", "An amount of 1200 is split in the ratio 2:3:5. What is the largest share?", "600"),
    Probe("hh39", "hard+", "exact", "What is the next number in the sequence 2, 6, 12, 20, 30?", "42"),
    Probe("hh40", "hard+", "exact", "If f(x) = 2x + 1 and g(x) = x squared, what is f(g(3)) minus g(f(3))?", "-30"),
]
