import bind_virtual_factory as bvf


class ImplClass(bvf.MyVirtualClass):
    def __init__(self):
        self.val = 42
        bvf.MyVirtualClass.__init__(self)

    def createData(self):
        return ImplData(self)

    # override MyVirtualClass::doSomethingPtr(shared_ptr data)
    def doSomethingPtr(self, data):
        print("Hello from doSomething!")
        assert isinstance(data, ImplData)
        print("Data value:", data.value)
        data.value += 1

    # override MyVirtualClass::doSomethingPtr(data&)
    def doSomethingRef(self, data):
        print("Hello from doSomethingRef!")
        print(type(data))
        assert isinstance(data, ImplData)
        print("Data value:", data.value)
        data.value += 1


class ImplData(bvf.MyVirtualData):
    def __init__(self, c):
        # parent virtual class requires arg
        bvf.MyVirtualData.__init__(self, c)
        self.value = c.val


def test_instantiate_child():
    obj = ImplClass()
    data = obj.createData()
    print(data)


def test_call_do_something_ptr():
    obj = ImplClass()
    print("Calling doSomething (by ptr)")
    d1 = bvf.callDoSomethingPtr(obj)
    print("Output data.value:", d1.value)


def test_call_do_something_ref():
    obj = ImplClass()
    print("Ref variant:")
    d2 = bvf.callDoSomethingRef(obj)
    print(d2.value)
    print("-----")


def test_iden_fns():
    obj = ImplClass()
    d = obj.createData()
    print(d, type(d))

    # take and return const T&
    d1 = bvf.iden_ref(d)
    print(d1, type(d1))
    assert isinstance(d1, ImplData)

    # take a shared_ptr, return const T&
    d2 = bvf.iden_shared(d)
    assert isinstance(d2, ImplData)
    print(d2, type(d2))

    print("copy shared ptr -> py -> cpp")
    d3 = bvf.copy_shared(d)
    assert isinstance(d3, ImplData)
    print(d3, type(d3))


test_instantiate_child()
test_call_do_something_ptr()
test_call_do_something_ref()
test_iden_fns()
