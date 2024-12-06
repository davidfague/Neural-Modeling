# calculates how the model deviations from the following assumptions:
# 1. at every branching point, the rule d_parent^(3/2) = sum(d_child^(3/2)) across children branches
# for 1 find every branching point and sum the differences at each branching point: d_parent^(3/2) - sum(d_child^(3/2)) 
# 2. every terminal section in a subtree (off of soma) has the same electrotonic distance
# for 2 calculate electrotonic length from soma to every terminal end, group by soma children, sum the difference from the maximum electrotonic length of the group.

# This computation represents reality's deviation from the ideal.