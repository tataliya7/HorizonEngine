#pragma once

namespace Ecila
{
    template<class T>
    class Singleton
    {
    private:

        Singleton(const Singleton<T>&) = delete;

        Singleton& operator=(const Singleton<T>&) = delete;

    protected:

        static T* mInstance;

        Singleton(void)
        {
            assert(!mInstance);
            mInstance = static_cast<T*>(this);
        }

        ~Singleton()
        {
            assert(mInstance);
            mInstance = nullptr;
        }

    public:

        static T* GetInstance()
        {
            return &mInstance;
        }
    };
}
