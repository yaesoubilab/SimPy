import SimPy.FormatFunctions as Format

print("\nFormatting estimates:")
print(Format.format_number(12345.678,deci=1))
print(Format.format_number(12345.678, deci=1, form=Format.FormatNumber.NUMBER))
print(Format.format_number(12345.678, deci=1, form=Format.FormatNumber.CURRENCY))
print(Format.format_number(0.12345, deci=1, form=Format.FormatNumber.PERCENTAGE))

print("\nFormatting intervals:")
print(Format.format_interval([12345.678, 98765.432],deci=1))
print(Format.format_interval([12345.678, 98765.432], deci=1, form=Format.FormatNumber.NUMBER))
print(Format.format_interval([12345.678, 98765.432], deci=1, form=Format.FormatNumber.CURRENCY))
print(Format.format_interval([0.12345, 0.98765], deci=1, form=Format.FormatNumber.PERCENTAGE))

print("\nFormatting estimates and intervals:")
print(Format.format_estimate_interval(
    estimate=5555.55, interval=[12345.678, 98765.432],deci=1))
print(Format.format_estimate_interval(
    estimate=5555.55, interval=[12345.678, 98765.432], deci=1, form=Format.FormatNumber.NUMBER))
print(Format.format_estimate_interval(
    estimate=5555.55, interval=[12345.678, 98765.432], deci=1, form=Format.FormatNumber.CURRENCY))
print(Format.format_estimate_interval(
    estimate=0.5555, interval=[0.12345, 0.98765], deci=1, form=Format.FormatNumber.PERCENTAGE))
