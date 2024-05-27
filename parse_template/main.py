import os
import sys
def parse_nested_structure(structure):
    stack = []
    current_item = ''
    for char in structure:
        if char == '<':
            if current_item:
                stack.append(current_item)
                current_item = ''
            stack.append('<')
        elif char == '>':
            if current_item:
                stack.append(current_item)
                current_item = ''
            items = []
            while stack[-1] != '<':
                items.append(stack.pop())
            stack.pop()  # remove '<'
            stack.append(items[::-1])
        elif char in ',':
            if current_item:
                stack.append(current_item)
                current_item = ''
        else:
            if char != ' ':
                current_item += char
    if current_item:
        stack.append(current_item)
    return stack
    
def print_nested_structure(structure, indent: int = 0):
    prefix = ' ' * indent
    if isinstance(structure, list):
        print(prefix+'<', end='\n')
        for i, item in enumerate(structure):
            print_nested_structure(item, indent + 4)
        print(prefix+'>', end='\n')
    else:
        print(prefix+structure, end='\n')
# parse the message
def parse(message: str):
    # parse the message, extract the template arguments. $template_function [with $template_names=$template_values. ]
    # eg: void cute::copy_unpack(const cute::Copy_Traits<cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<cutlass::uint128_t, cutlass::uint128_t>> &, const cute::Tensor<TS, SLayout> &, cute::Tensor<TD, DLayout> &) [with TS=cute::ViewEngine<cute::gmem_ptr<int8_t *>>, SLayout=cute::Layout<cute::tuple<cute::tuple<cute::_16, cute::_1>>, cute::tuple<cute::tuple<int, cute::_0>>>, TD=cute::ViewEngine<cute::smem_ptr<int8_t *>>, DLayout=cute::Layout<cute::tuple<cute::tuple<cute::_16, cute::_1>>, cute::tuple<cute::tuple<cute::C<64>, cute::_0>>>]
    # extract $template_function, it seperated with [with
    template_function,rest = message.split("[with")[:2]
    print(template_function)
    
    # ignore "]"
    rest = rest[:-1]
    # print(rest)
    # extract $template_names
    # $item=$template_names=$template_values
    # rest = [$item].
    # each $item is a kv pair, k=v
    # parse rest into list of $item, using "="
    list_of_items = rest.split("=") #(k_0,v_0k_1,v_1+k_2,v_2+k_3,...,v_n)
    kv_pairs = [list_of_items[0]]
    for item in list_of_items[1:-1]:
        *v,k = item.split(",")
        v = ",".join(v)
        kv_pairs.append(v)
        kv_pairs.append(k)
    kv_pairs.append(list_of_items[-1])
    ks = kv_pairs[::2]
    vs = kv_pairs[1::2]
    parse_vs = [parse_nested_structure(v) for v in vs]
    print(parse_vs)
    for k,v in zip(ks,parse_vs):
        print(k)
        print_nested_structure(v)
        print()
    # parse vs
    # vs = str | <vs>

    
    
if __name__ == '__main__':
    message = sys.argv[1]
    parse(message)
    