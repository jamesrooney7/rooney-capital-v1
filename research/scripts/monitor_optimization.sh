#!/bin/bash
while true; do
  clear
  echo "=== ML Optimization Progress ==="
  echo "$(date '+%Y-%m-%d %H:%M:%S')"
  echo ""

  for dir in results/*_optimization; do
    if [ ! -d "$dir" ]; then continue; fi

    symbol=$(basename "$dir" | sed 's/_optimization//')

    # Get counts, ensure they're single integers
    rs_count=$(grep -c "^\[RS" "$dir/optimization_${symbol}.log" 2>/dev/null || echo "0")
    bo_count=$(grep -c "^\[BO" "$dir/optimization_${symbol}.log" 2>/dev/null || echo "0")

    # Strip any whitespace/newlines and default to 0 if empty
    rs_count=$(echo "$rs_count" | tr -d '\n\r' | grep -o '[0-9]*' | head -1)
    bo_count=$(echo "$bo_count" | tr -d '\n\r' | grep -o '[0-9]*' | head -1)
    rs_count=${rs_count:-0}
    bo_count=${bo_count:-0}

    total=$((rs_count + bo_count))

    # Get best Sharpe
    best_sr=$(grep -E "^\[RS|^\[BO" "$dir/optimization_${symbol}.log" 2>/dev/null | \
              grep -oP "SR=\K[0-9.]+" | sort -rn | head -1 || echo "N/A")

    # Progress bar
    progress=$((total * 100 / 90))
    bar_length=20
    filled=$((progress * bar_length / 100))
    bar=$(printf "%${filled}s" | tr ' ' '█')
    empty=$(printf "%$((bar_length - filled))s" | tr ' ' '░')

    # Status
    if [ $total -eq 90 ]; then
      status="✓ DONE"
    elif [ $bo_count -gt 0 ]; then
      status="BO"
    elif [ $rs_count -gt 0 ]; then
      status="RS"
    else
      status="Starting"
    fi

    printf "%-4s [%s%s] %3d%% (%2d/90) | Best SR: %-6s | %s\n" \
           "$symbol" "$bar" "$empty" "$progress" "$total" "$best_sr" "$status"
  done

  echo ""
  echo "Running processes: $(ps aux | grep rf_cpcv_random_then_bo.py | grep -v grep | wc -l)/16"
  echo ""
  echo "Press Ctrl+C to exit"

  sleep 30
done
