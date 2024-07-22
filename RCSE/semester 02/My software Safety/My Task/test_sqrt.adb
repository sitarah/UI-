with Ada.Text_IO; use Ada.Text_IO;
with Arith;

procedure Test_Sqrt is
    Arg    : Integer := 25; --  different input values
    Result : Integer;
begin
    Result := Arith.Sqrt(Arg);
    Put_Line("The square root of " & Integer'Image(Arg) & " is " & Integer'Image(Result));
end Test_Sqrt.
