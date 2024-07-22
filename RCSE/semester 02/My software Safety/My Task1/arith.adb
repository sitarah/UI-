package body arith is
   function Sqroot (Arg : Integer) return Integer is
      Result : Integer;
   begin
      Result := Integer(Sqrt(Arg));
      Put_Line("Calculating Square root of " & Arg'Image);
      Put_Line("The result is " & Result'Image);
      return Result;
   end Sqroot;
end arith;
