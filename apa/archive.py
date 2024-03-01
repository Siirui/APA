from augmentations import augment_list

def remove_deplicates(policies):
    s = set()
    new_policies = []
    for ops in policies:
        key = []
        for op in ops:
            key.append(op[0])
        key = '_'.join(key)
        if key in s:
            continue
        else:
            s.add(key)
            new_policies.append(ops)

    return new_policies


def policy_decoder(policy, num_subpolicy, num_op):
    op_list = augment_list()
    decoded_policy = []
    for i in range(num_subpolicy):
        subpolicy = []
        for j in range(num_op):
            op_idx = policy['policy_%d_%d' % (i, j)]
            prob = policy['prob_%d_%d' % (i, j)]
            level = policy['level_%d_%d' % (i, j)]
            subpolicy.append((op_list[op_idx][0].__name__, prob, level))
        decoded_policy.append(subpolicy)
    return decoded_policy
