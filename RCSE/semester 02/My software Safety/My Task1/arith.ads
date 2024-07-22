package arith is
   pragma Assertion_Policy (Check);
   function Sqroot (Arg : Integer) return Integer
     with Pre => Arg >= 0,
          Post => (Sqroot'Result >= 0)
                   and ((Sqroot'Result ** 2) <= Arg)
                   and ((Sqroot'Result ** 2 + 1) > Arg);
end arith;
