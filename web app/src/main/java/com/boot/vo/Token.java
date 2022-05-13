package com.boot.vo;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * 用户登录通信令牌
 */
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Token {
	private Integer businessId, seriaNo;
	private String username,name;
	private Long loginTime;
	private String userId;
}
