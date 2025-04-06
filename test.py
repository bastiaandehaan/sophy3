import vectorbt as vbt

# Test 1: Direct .set() method
try:
    vbt.settings.set(portfolio={"fees": 0.0001})
    print("✅ Direct .set() method works")
except Exception as e:
    print(f"❌ Direct .set() method failed: {e}")

# Test 2: Alternative configuration
try:
    vbt.settings.portfolio["fees"] = 0.0001
    print("✅ Direct attribute assignment works")
except Exception as e:
    print(f"❌ Direct attribute assignment failed: {e}")

# Test 3: Checking current settings
print("\nCurrent VectorBT Portfolio Settings:")
print(vbt.settings.portfolio)