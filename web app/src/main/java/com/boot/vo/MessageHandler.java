package com.boot.vo;

/**
 * 生成消息
 */
public class MessageHandler {
	/**
	 * 创建一个请求失败的消息类
	 */
	public static BaseMessage<String> createFailedVo(String msg) {
		return new BaseMessage<String>(false, null, msg,null,0);
	}
	/**
	 * 创建一个请求失败的消息类 code
	 * @return
	 */
	public static BaseMessage<String> createFailedVo(String msg,Integer code) {
		return new BaseMessage<String>(false, null, msg,null,code);
	}

	/**
	 * 创建一个请求成功的消息类
	 */
	public static BaseMessage<String> createSuccessVo(String msg) {
		return new BaseMessage<String>(true, null, msg,null,0);
	}

	/**
	 * 创建一个请求成功的带数据的消息类
	 */
	public static <T> BaseMessage<T> createSuccessVo(T t, String msg) {
		return new BaseMessage<T>(true, t, msg,null,0);
	}
	public static <T> BaseMessage<T> createSuccessVo(T t, String msg,Integer count) {
		return new BaseMessage<T>(true, t, msg,count,0);
	}
}
