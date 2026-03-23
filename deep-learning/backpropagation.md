# ⬅️ Backpropagation

> האלגוריתם שמחשב את הגרדיאנט של ה-loss לכל משקל ברשת — על ידי כלל השרשרת מהפלט לכניסה.

---

## 4 השלבים

| שלב | מה קורה | נוסחה |
|-----|---------|-------|
| 1. Forward | חשב פלט כל שכבה | `h = f(Wx + b)` |
| 2. Loss | מדד שגיאה | `L = ½‖y - ŷ‖²` |
| 3. Backward | גרדיאנט לכל שכבה | `∂L/∂W` via chain rule |
| 4. Update | עדכן משקלים | `W ← W - η·∂L/∂W` |

---

## נוסחאות מרכזיות

$$W_{new} = W_{old} - \eta \cdot \frac{\partial L}{\partial W}$$

$$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial h} \cdot \frac{\partial h}{\partial W_1}$$

---

## קוד

```python
# גרדיאנט ידני לשכבה אחת
def backward(dL_dy, x, W):
    dL_dW = x.T @ dL_dy         # ∂L/∂W
    dL_db = dL_dy.sum(axis=0)   # ∂L/∂b
    dL_dx = dL_dy @ W.T         # להעביר לשכבה הקודמת
    return dL_dW, dL_db, dL_dx

# PyTorch עושה הכל אוטומטית:
loss.backward()   # מחשב כל הגרדיאנטים
optimizer.step()  # מעדכן W בכל שכבות
```

---

## Vanishing Gradient

> ⚠️ ברשתות עמוקות הגרדיאנט מתכווץ ל-0 — השכבות הראשונות לא לומדות.

**פתרונות:**
- **ReLU** במקום Sigmoid
- **BatchNormalization**
- **Skip Connections** (ResNet)
- **Gradient Clipping**
