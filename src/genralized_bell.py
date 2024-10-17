import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

x = np.linspace(0, 10, 100)

generalized_bell = fuzz.gbellmf(x, a=2, b=4, c=5)

# Візуалізація
plt.plot(x, generalized_bell, label='Узагальнений дзвін')
plt.title('Узагальнений дзвін')
plt.legend()
plt.show()
