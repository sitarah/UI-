package Arith is

    -- Sqrt calculates the square root of the provided integer argument
    -- Preconditions:
    --   - Argument must be positive
    -- Postconditions:
    --   - Result is non-negative
    --   - The square of the result is less than or equal to the argument
    --   - The square of the result + 1 is greater than the argument
    function Sqrt (Arg : Integer) return Integer
      with Pre  => Arg > 0,
           Post => (Sqrt'Result >= 0) and then
                   (Sqrt'Result * Sqrt'Result <= Arg) and then
                   ((Sqrt'Result + 1) * (Sqrt'Result + 1) > Arg);
end Arith;
