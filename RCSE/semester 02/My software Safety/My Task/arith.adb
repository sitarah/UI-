with Ada.Text_IO; use Ada.Text_IO;

package body Arith is

    function Sqrt (Arg : Integer) return Integer is
        -- Starting with an initial Estimation
        Estimation     : Integer := Arg / 2;
        Previous  : Integer := 0;
    begin
        Put_Line("Calculating Sqrt of " & Arg'Image);

        -- Special case for Arg = 1
        if Arg = 1 then
            Put_Line("The result is 1");
            return 1;
        end if;

        -- Iteratively improve the Estimation
        while Estimation /= Previous loop
            Previous := Estimation;
            Estimation := (Estimation + Arg / Estimation) / 2;
        end loop;

        Put_Line("The result is " & Estimation'Image);

        -- Return the integer part of the square root
        return Estimation;
    end Sqrt;

end Arith;
