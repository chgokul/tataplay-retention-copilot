-- Q1: successful recharge revenue by plan
SELECT p.plan_name, SUM(r.amount_inr) AS revenue_inr,
       ROUND(AVG(r.amount_inr),2) AS avg_recharge_inr,
       COUNT(DISTINCT r.subscriber_id) AS subscribers
FROM recharges r
JOIN subscribers s USING(subscriber_id)
JOIN plans p USING(plan_id)
WHERE LOWER(r.status)='success'
GROUP BY p.plan_name
ORDER BY revenue_inr DESC;

-- Q2: subscribers with no successful recharge (LEFT JOIN preserves them)
SELECT s.subscriber_id, s.tenure_months
FROM subscribers s LEFT JOIN recharges r
ON s.subscriber_id=r.subscriber_id AND LOWER(r.status)='success'
WHERE r.recharge_id IS NULL;

-- Q3: totals and all three ranking functions within city
WITH totals AS (
 SELECT s.city, s.subscriber_id, SUM(r.amount_inr) AS total
 FROM subscribers s JOIN recharges r USING(subscriber_id)
 WHERE LOWER(r.status)='success'
 GROUP BY s.city,s.subscriber_id
)
SELECT city, subscriber_id, total,
 RANK() OVER(PARTITION BY city ORDER BY total DESC) AS rnk,
 DENSE_RANK() OVER(PARTITION BY city ORDER BY total DESC) AS dense_rnk,
 ROW_NUMBER() OVER(PARTITION BY city ORDER BY total DESC, subscriber_id) AS row_num
FROM totals
ORDER BY city, row_num;

-- Q4: running successful recharge total for TP00010
SELECT recharge_id, recharge_date, amount_inr,
 SUM(amount_inr) OVER(ORDER BY recharge_date,recharge_id) AS running_total
FROM recharges
WHERE subscriber_id='TP00010' AND LOWER(status)='success'
ORDER BY recharge_date,recharge_id;
