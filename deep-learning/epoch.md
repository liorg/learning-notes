# 🔄 Epoch & Training Loop

> Epoch = מעבר אחד מלא על כל ה-dataset. כל epoch מחולק ל-batches.

---

## מושגי מפתח

| מושג | הגדרה | ערך טיפוסי |
|------|-------|-----------|
| `Epoch` | מעבר מלא על כל ה-dataset | 10–200 |
| `Batch Size` | כמה דוגמאות בכל עדכון | 32, 64, 128 |
| `Iteration` | עדכון משקלים אחד | N / batch_size |
| `Learning Rate η` | גודל הצעד בגרדיאנט | 0.001 – 0.01 |
| `Loss` | כמה טועה המודל | → 0 עם הזמן |

---

## Training Loop

```
Dataset → Batch → Forward Pass → Loss → Backward Pass → Update W → חזור
```

```python
for epoch in range(100):
    for X_batch, y_batch in dataloader:
        # 1. Forward pass
        y_pred = model(X_batch)
        loss   = criterion(y_pred, y_batch)

        # 2. Backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
```

---

## Overfitting vs Underfitting

| מצב | סימנים | פתרון |
|-----|--------|-------|
| **Overfitting** | train loss ↓, val loss ↑ | Dropout, Regularization, יותר data |
| **Underfitting** | שניהם גבוהים | רשת גדולה יותר, יותר epochs |
| **תקין** | שניהם יורדים יחד | 🎉 |

> ⚠️ **Learning Rate גדול מדי** → Loss קופץ ולא מתכנס  
> ⚠️ **Learning Rate קטן מדי** → אימון איטי מאוד
