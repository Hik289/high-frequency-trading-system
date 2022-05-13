package com.boot.vo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 消息类，具体使用可以参照{@link MessageHandler}
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class BaseMessage<T> {
	private boolean success;
	private T data;
	private String message;
	private Integer count,code;

}
